#!/usr/bin/env python3
"""Train, evaluate, visualize, and compare every supported Stage 1 encoder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageDraw

from tomato_recon.data.tomatowur import ProcessedTomatoDataset
from tomato_recon.models.encoders.registry import ensure_backbone_available
from tomato_recon.models.pretrained import verify_sonata_checkpoint

MODELS = ("pointnext", "sonata_ptv3", "kpconvx")
EVENT_PREFIX = "@@STAGE1_EVENT@@"


def _duration(seconds: float | None) -> str:
    if seconds is None or not torch.isfinite(torch.tensor(seconds)):
        return "--:--:--"
    seconds_int = max(int(seconds), 0)
    minutes, seconds_int = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}"


class AblationProgress:
    """Global, monotonic Stage 1 progress with a durable JSONL event stream."""

    def __init__(self, output: Path, model_count: int) -> None:
        self.output = output
        self.model_count = model_count
        self.started = time.perf_counter()
        self.last_percent = 0.0
        output.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        model_index: int,
        model: str,
        phase: str,
        model_fraction: float,
        detail: str,
        *,
        status: str = "running",
        **extra: object,
    ) -> None:
        percent = 100.0 * (model_index + min(max(model_fraction, 0.0), 1.0)) / self.model_count
        percent = max(percent, self.last_percent)
        self.last_percent = percent
        elapsed = time.perf_counter() - self.started
        eta = elapsed * (100.0 - percent) / percent if percent > 0 else None
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "model_index": model_index + 1,
            "model_count": self.model_count,
            "phase": phase,
            "status": status,
            "overall_percent": percent,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            **extra,
        }
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        print(
            f"[Stage1 ablation {percent:5.1f}%] model {model_index + 1}/{self.model_count} "
            f"{model} | {detail} | elapsed {_duration(elapsed)} | ETA {_duration(eta)}",
            flush=True,
        )


def _run(command: list[str], event_handler: Callable[[dict[str, Any]], None] | None = None) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["STAGE1_ABLATION_EVENTS"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith(EVENT_PREFIX):
            if event_handler:
                event_handler(json.loads(line[len(EVENT_PREFIX) :]))
            continue
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _visualizations_complete(path: Path, expected_ids: list[str]) -> bool:
    manifest_path = path / "visualization_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    renders = manifest.get("renders", {})
    return all(
        plant_id in renders
        and renders[plant_id].get("status") == "complete"
        and Path(renders[plant_id]["path"]).is_file()
        for plant_id in expected_ids
    )


def _chart(path: Path, title: str, values: dict[str, list[tuple[str, float]]]) -> None:
    colours = {"pointnext": (60, 120, 210), "sonata_ptv3": (50, 165, 95), "kpconvx": (220, 125, 45)}
    width = 1000
    row_height = 42
    height = 90 + row_height * sum(len(items) + 1 for items in values.values())
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title, fill=(20, 20, 20))
    y = 55
    for group, items in values.items():
        draw.text((20, y), group, fill=(40, 40, 40))
        y += row_height
        for model, value in items:
            bar_start = 190
            bar_width = int(max(0.0, min(value, 1.0)) * 700)
            draw.text((35, y + 8), model, fill=(45, 45, 45))
            draw.rectangle(
                (bar_start, y + 7, bar_start + bar_width, y + 29),
                fill=colours.get(model, (100, 100, 100)),
            )
            draw.text((bar_start + bar_width + 8, y + 8), f"{value:.4f}", fill=(30, 30, 30))
            y += row_height
        y += 6
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_comparison(output: Path, results: dict[str, dict[str, Any]]) -> None:
    successful = {
        model: result for model, result in results.items() if result.get("status") == "complete"
    }
    ranked = sorted(
        successful,
        key=lambda model: successful[model]["validation"]["metrics"]["overall_score"],
        reverse=True,
    )
    report = {
        "schema_version": "1.0",
        "selection_split": "val",
        "test_used_for_selection": False,
        "ranking": ranked,
        "winner": ranked[0] if ranked else None,
        "models": results,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fields = [
        "model",
        "status",
        "split",
        "overall_score",
        "semantic_miou",
        "skeleton_precision",
        "skeleton_recall",
        "skeleton_f1",
        "centreline_offset_mae_mm",
        "centreline_offset_score",
        "junction_precision",
        "junction_recall",
        "junction_f1",
        "semantic_loss",
        "skeleton_loss",
        "offset_loss",
        "junction_loss",
        "parameter_count",
        "trainable_parameter_count",
        "best_epoch",
        "training_runtime_seconds",
        "peak_gpu_memory_mb",
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            result = results.get(model, {"status": "not_run"})
            if result.get("status") != "complete":
                writer.writerow({"model": model, "status": result.get("status", "not_run")})
                continue
            training = result.get("training", {})
            for split_key in ("validation", "test"):
                metrics = result[split_key]["metrics"]
                writer.writerow(
                    {
                        "model": model,
                        "status": "complete",
                        "split": result[split_key]["split"],
                        **{key: metrics.get(key) for key in fields if key in metrics},
                        **{key: training.get(key) for key in fields if key in training},
                    }
                )

    lines = [
        "# Stage 1 encoder ablation",
        "",
        "Models are ranked by validation overall score. Test metrics are held out and were not used for selection.",
        "",
        "| Rank | Model | Val overall | Test overall | Semantic mIoU | Skeleton F1 | Offset MAE (mm) | Junction F1 | Visualizations |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, model in enumerate(ranked, start=1):
        result = successful[model]
        val = result["validation"]["metrics"]
        test = result["test"]["metrics"]
        lines.append(
            f"| {rank} | {model} | {val['overall_score']:.4f} | {test['overall_score']:.4f} "
            f"| {test['semantic_miou']:.4f} | {test['skeleton_f1']:.4f} "
            f"| {test['centreline_offset_mae_mm']:.2f} | {test['junction_f1']:.4f} "
            f"| [{model} test set]({model}/test_visualizations/) |"
        )
    failed = [model for model in MODELS if model not in successful]
    if failed:
        lines.extend(["", "## Incomplete models", ""])
        for model in failed:
            lines.append(f"- `{model}`: {results.get(model, {}).get('error', 'not run')}")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if successful:
        _chart(
            output / "overall_scores.png",
            "Stage 1 overall scores (validation and held-out test)",
            {
                "Validation": [(model, successful[model]["validation"]["metrics"]["overall_score"]) for model in MODELS if model in successful],
                "Test": [(model, successful[model]["test"]["metrics"]["overall_score"]) for model in MODELS if model in successful],
            },
        )
        task_groups: dict[str, list[tuple[str, float]]] = {}
        for label, key in (
            ("Semantic mIoU", "semantic_miou"),
            ("Skeleton F1", "skeleton_f1"),
            ("Offset score", "centreline_offset_score"),
            ("Junction F1", "junction_f1"),
        ):
            task_groups[label] = [
                (model, successful[model]["test"]["metrics"][key])
                for model in MODELS
                if model in successful
            ]
        _chart(output / "task_metrics.png", "Stage 1 held-out test task metrics", task_groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/v3_gt_K256"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1_ablation"))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Testing only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not (args.processed_root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"processed dataset missing at {args.processed_root}; run preprocessing first"
        )
    validation_dataset = ProcessedTomatoDataset(args.processed_root, split="val")
    test_dataset = ProcessedTomatoDataset(args.processed_root, split="test")
    if not len(validation_dataset) or not len(test_dataset):
        raise ValueError("Stage 1 ablation requires non-empty val and test splits")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Stage 1 ablation requires a CUDA GPU visible inside Docker")
    verify_sonata_checkpoint("data/pretrained/sonata/sonata.pth")
    for model in MODELS:
        ensure_backbone_available(model)

    expected_test_ids = [test_dataset[index].plant_id for index in range(len(test_dataset))]
    progress = AblationProgress(args.output / "progress.jsonl", len(MODELS))
    results: dict[str, dict[str, Any]] = {}
    failures = 0

    for model_index, model in enumerate(MODELS):
        model_output = args.output / model
        complete_path = model_output / "run_complete.json"
        if args.resume and complete_path.is_file():
            result = _read_json(complete_path)
            results[model] = result
            progress.emit(
                model_index, model, "complete", 1.0, "complete (resume: skipped)", status="skipped"
            )
            continue
        model_output.mkdir(parents=True, exist_ok=True)
        try:
            training_marker = model_output / "training_complete.json"
            checkpoint = model_output / "best.ckpt"
            if not (args.resume and training_marker.is_file() and checkpoint.is_file()):
                progress.emit(model_index, model, "train", 0.0, f"train epoch 0/{args.max_epochs}")
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "tomato_recon.train.train_encoder",
                    "--config",
                    f"configs/encoder/{model}.yaml",
                    f"data.processed_root={args.processed_root}",
                    f"output.dir={model_output}",
                    f"trainer.max_epochs={args.max_epochs}",
                    "trainer.batch_size=1",
                    "trainer.num_workers=0",
                    "seed=42",
                ]
                if args.allow_cpu:
                    command.append("trainer.devices=0")
                if args.resume and (model_output / "last.ckpt").is_file():
                    command.extend(["--resume", str(model_output / "last.ckpt")])

                def training_event(event: dict[str, Any]) -> None:
                    epoch = int(event["epoch"])
                    epoch_total = int(event["epoch_total"])
                    progress.emit(
                        model_index,
                        model,
                        "train",
                        0.90 * epoch / max(epoch_total, 1),
                        f"train epoch {epoch}/{epoch_total}",
                        epoch=epoch,
                        epoch_total=epoch_total,
                    )

                _run(command, training_event)
                training_marker.write_text(
                    json.dumps({"status": "complete", "checkpoint": str(checkpoint)}, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
            else:
                progress.emit(model_index, model, "train", 0.90, "training complete (resume: skipped)")

            validation_path = model_output / "validation_metrics.json"
            test_path = model_output / "test_metrics.json"
            for split_index, (split, destination) in enumerate(
                (("val", validation_path), ("test", test_path)), start=1
            ):
                if not (args.resume and destination.is_file()):
                    _run(
                        [
                            sys.executable,
                            "-u",
                            "scripts/evaluate_encoder.py",
                            "--checkpoint",
                            str(checkpoint),
                            "--processed-root",
                            str(args.processed_root),
                            "--split",
                            split,
                            "--output",
                            str(destination),
                            "--device",
                            "cpu" if args.allow_cpu else "cuda",
                        ]
                    )
                progress.emit(
                    model_index,
                    model,
                    "evaluate",
                    0.90 + 0.025 * split_index,
                    f"evaluate {split} complete",
                    split=split,
                )

            visual_output = model_output / "test_visualizations"
            if not (args.resume and _visualizations_complete(visual_output, expected_test_ids)):
                progress.emit(
                    model_index,
                    model,
                    "visualize",
                    0.95,
                    f"visualize test plant 0/{len(expected_test_ids)}",
                )

                def visual_event(event: dict[str, Any]) -> None:
                    plant = int(event["plant"])
                    plant_total = int(event["plant_total"])
                    progress.emit(
                        model_index,
                        model,
                        "visualize",
                        0.95 + 0.05 * plant / max(plant_total, 1),
                        f"visualize test plant {plant}/{plant_total}",
                        plant=plant,
                        plant_total=plant_total,
                        plant_id=event.get("plant_id"),
                    )

                _run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/visualize_encoder_predictions.py",
                        "--checkpoint",
                        str(checkpoint),
                        "--processed-root",
                        str(args.processed_root),
                        "--split",
                        "test",
                        "--count",
                        "0",
                        "--output",
                        str(visual_output),
                        "--device",
                        "cpu" if args.allow_cpu else "cuda",
                    ],
                    visual_event,
                )
            if not _visualizations_complete(visual_output, expected_test_ids):
                raise RuntimeError("test visualization manifest is incomplete")

            training = _read_json(model_output / "metrics.json")
            result = {
                "status": "complete",
                "model": model,
                "checkpoint": str(checkpoint),
                "training": training,
                "validation": _read_json(validation_path),
                "test": _read_json(test_path),
                "test_visualizations": str(visual_output),
            }
            complete_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            results[model] = result
            val_score = result["validation"]["metrics"]["overall_score"]
            progress.emit(
                model_index,
                model,
                "complete",
                1.0,
                f"complete | val overall {val_score:.4f}",
                status="complete",
                validation_overall_score=val_score,
            )
        except Exception as exc:  # keep independent ablations running
            failures += 1
            result = {
                "status": "failed",
                "model": model,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results[model] = result
            (model_output / "failure.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            progress.emit(
                model_index,
                model,
                "failed",
                1.0,
                f"failed: {exc}",
                status="failed",
                error=str(exc),
            )
        _write_comparison(args.output, results)

    _write_comparison(args.output, results)
    if failures:
        raise SystemExit(f"Stage 1 ablation completed with {failures} failed model(s)")
    winner = _read_json(args.output / "comparison.json")["winner"]
    print(f"Stage 1 ablation complete; validation winner: {winner}", flush=True)


if __name__ == "__main__":
    main()
