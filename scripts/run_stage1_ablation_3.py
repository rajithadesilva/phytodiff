#!/usr/bin/env python3
"""Run Stage 1 Ablation 3 across training and evaluation datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

if __package__:
    from scripts.run_stage1_ablation_1 import _pcl_types_override, _read_json, _run
else:
    from run_stage1_ablation_1 import _pcl_types_override, _read_json, _run
from tomato_recon.data.processed import ProcessedPlantDataset
from tomato_recon.models.encoders.registry import ensure_backbone_available
from tomato_recon.train.common import checkpoint_sha256

MODEL = "kpconvx"
PCL_TYPES = ("full", "top_down", "side")
DATASETS = ("tomatowur", "tomatopgt", "pheno4d", "combined")
DATASET_LABELS = {
    "tomatowur": "TomatoWUR",
    "tomatopgt": "TomatoPGT",
    "pheno4d": "Pheno4D",
    "combined": "Combined",
}
MATRIX_METRICS: tuple[tuple[str, str], ...] = (
    ("semantic_miou", "Semantic mIoU"),
    ("skeleton_f1", "Skeleton F1"),
    ("centreline_offset_score", "Centreline offset score"),
    ("junction_f1", "Junction F1"),
    ("overall_score", "Weighted overall score"),
)
CSV_METRICS = (
    "loss",
    "semantic_miou",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "centreline_offset_mae_m",
    "centreline_offset_mae_mm",
    "centreline_offset_score",
    "junction_precision",
    "junction_recall",
    "junction_f1",
    "overall_score",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1_ablation_3"))
    parser.add_argument("--max-epochs", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Testing only")
    return parser


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    seconds_int = max(int(seconds), 0)
    minutes, seconds_int = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}"


class Ablation3Progress:
    """Durable progress across four training runs and sixteen evaluations."""

    def __init__(self, output: Path, run_count: int) -> None:
        self.output = output
        self.run_count = run_count
        self.started = time.perf_counter()
        self.last_percent = 0.0
        output.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        run_index: int,
        training_dataset: str,
        evaluation_dataset: str | None,
        phase: str,
        run_fraction: float,
        detail: str,
        *,
        status: str = "running",
        **extra: object,
    ) -> None:
        percent = 100.0 * (
            run_index + min(max(run_fraction, 0.0), 1.0)
        ) / self.run_count
        percent = max(percent, self.last_percent)
        self.last_percent = percent
        elapsed = time.perf_counter() - self.started
        eta = elapsed * (100.0 - percent) / percent if percent > 0 else None
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training_dataset": training_dataset,
            "evaluation_dataset": evaluation_dataset,
            "run_index": run_index + 1,
            "run_count": self.run_count,
            "phase": phase,
            "status": status,
            "overall_percent": percent,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            **extra,
        }
        with self.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        pair = (
            training_dataset
            if evaluation_dataset is None
            else f"{training_dataset}->{evaluation_dataset}"
        )
        print(
            f"[Stage 1 Ablation 3 {percent:5.1f}%] run {run_index + 1}/"
            f"{self.run_count} {pair} | {detail} | elapsed {_duration(elapsed)} | "
            f"ETA {_duration(eta)}",
            flush=True,
        )


def _require_training_marker(
    marker: dict[str, Any],
    *,
    dataset: str,
    batch_size: int,
    checkpoint: Path,
) -> None:
    expected = {
        "model": MODEL,
        "dataset": dataset,
        "pcl_types": list(PCL_TYPES),
        "batch_size": batch_size,
    }
    actual = {key: marker.get(key) for key in expected}
    if marker.get("status") != "complete" or actual != expected:
        raise ValueError(
            f"stale Ablation 3 training record at {checkpoint.parent}: "
            f"expected {expected}, got {actual}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ablation 3 checkpoint is missing: {checkpoint}")
    checksum = checkpoint_sha256(checkpoint)
    if marker.get("checkpoint_sha256") != checksum:
        raise ValueError(f"Ablation 3 checkpoint checksum changed: {checkpoint}")


def _require_resume_checkpoint(
    checkpoint: Path,
    *,
    dataset: str,
    batch_size: int,
) -> None:
    """Reject a partial checkpoint produced by a different experiment setup."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    data = config.get("data", {}) if isinstance(config, dict) else {}
    trainer = config.get("trainer", {}) if isinstance(config, dict) else {}
    model = config.get("model", {}) if isinstance(config, dict) else {}
    encoder = model.get("encoder", {}) if isinstance(model, dict) else {}
    expected = {
        "stage": "encoder",
        "model": MODEL,
        "dataset": dataset,
        "pcl_types": list(PCL_TYPES),
        "batch_size": batch_size,
    }
    actual = {
        "stage": payload.get("stage"),
        "model": encoder.get("name"),
        "dataset": data.get("dataset"),
        "pcl_types": data.get("pcl_types"),
        "batch_size": trainer.get("batch_size"),
    }
    if actual != expected:
        raise ValueError(
            f"stale Ablation 3 resume checkpoint at {checkpoint}: "
            f"expected {expected}, got {actual}"
        )


def _require_evaluation_marker(
    marker: dict[str, Any],
    *,
    training_dataset: str,
    evaluation_dataset: str,
    batch_size: int,
    checkpoint_sha: str,
) -> None:
    expected = {
        "model": MODEL,
        "training_dataset": training_dataset,
        "evaluation_dataset": evaluation_dataset,
        "pcl_types": list(PCL_TYPES),
        "batch_size": batch_size,
        "checkpoint_sha256": checkpoint_sha,
    }
    actual = {key: marker.get(key) for key in expected}
    if marker.get("status") != "complete" or actual != expected:
        raise ValueError(
            "stale Ablation 3 evaluation record: "
            f"expected {expected}, got {actual}"
        )


def _metric_matrix(
    evaluations: dict[str, dict[str, Any]], metric: str
) -> list[list[float | None]]:
    matrix: list[list[float | None]] = []
    for training_dataset in DATASETS:
        row: list[float | None] = []
        for evaluation_dataset in DATASETS:
            result = evaluations.get(
                f"{training_dataset}->{evaluation_dataset}", {}
            )
            value = result.get("metrics", {}).get(metric)
            row.append(float(value) if value is not None else None)
        matrix.append(row)
    return matrix


def _heatmap_colour(value: float | None) -> tuple[int, int, int]:
    if value is None or not math.isfinite(value):
        return (220, 220, 220)
    clipped = min(max(value, 0.0), 1.0)
    return (
        int(225 - 150 * clipped),
        int(85 + 135 * clipped),
        int(75 + 55 * clipped),
    )


def write_heatmap(
    path: Path,
    title: str,
    matrix: list[list[float | None]],
) -> None:
    """Write a dependency-light 4x4 metric heatmap with explicit axes."""
    cell_width = 145
    cell_height = 62
    left = 170
    top = 105
    width = left + cell_width * len(DATASETS) + 25
    height = top + cell_height * len(DATASETS) + 65
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((18, 15), title, fill=(20, 20, 20))
    draw.text((18, 36), "Rows: checkpoint training dataset", fill=(55, 55, 55))
    draw.text((18, 52), "Columns: held-out test dataset", fill=(55, 55, 55))

    for column, dataset in enumerate(DATASETS):
        label = DATASET_LABELS[dataset]
        center_x = left + column * cell_width + cell_width / 2
        box = draw.textbbox((0, 0), label)
        draw.text((center_x - (box[2] - box[0]) / 2, 77), label, fill=(40, 40, 40))
    for row, training_dataset in enumerate(DATASETS):
        row_y = top + row * cell_height
        draw.text((12, row_y + 23), DATASET_LABELS[training_dataset], fill=(40, 40, 40))
        for column, value in enumerate(matrix[row]):
            x0 = left + column * cell_width
            draw.rectangle(
                (x0 + 1, row_y + 1, x0 + cell_width - 2, row_y + cell_height - 2),
                fill=_heatmap_colour(value),
                outline=(245, 245, 245),
            )
            text_value = "failed" if value is None else f"{value:.4f}"
            box = draw.textbbox((0, 0), text_value)
            draw.text(
                (x0 + (cell_width - (box[2] - box[0])) / 2, row_y + 23),
                text_value,
                fill=(25, 25, 25),
            )
    draw.text((left, height - 35), "grey = unavailable", fill=(70, 70, 70))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_ablation_3_report(
    output: Path,
    *,
    batch_size: int,
    training_results: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
) -> None:
    """Write machine-readable results and one heatmap per Stage 1 task."""
    matrices = {
        metric: _metric_matrix(evaluations, metric) for metric, _ in MATRIX_METRICS
    }
    report = {
        "schema_version": "1.0",
        "experiment": "stage1_ablation_3",
        "model": MODEL,
        "pcl_types": list(PCL_TYPES),
        "batch_size": batch_size,
        "selection_split": "val",
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "dataset_order": list(DATASETS),
        "training_runs": training_results,
        "evaluations": evaluations,
        "matrices": matrices,
    }
    (output / "matrix.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fields = [
        "training_dataset",
        "evaluation_dataset",
        "model",
        "pcl_types",
        "batch_size",
        "status",
        "checkpoint",
        "checkpoint_sha256",
        "sample_count",
        "view_sample_counts",
        *CSV_METRICS,
        "error",
    ]
    with (output / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for training_dataset in DATASETS:
            for evaluation_dataset in DATASETS:
                result = evaluations.get(
                    f"{training_dataset}->{evaluation_dataset}",
                    {"status": "not_run"},
                )
                metrics = result.get("metrics", {})
                writer.writerow(
                    {
                        "training_dataset": training_dataset,
                        "evaluation_dataset": evaluation_dataset,
                        "model": MODEL,
                        "pcl_types": " ".join(PCL_TYPES),
                        "batch_size": batch_size,
                        "status": result.get("status", "not_run"),
                        "checkpoint": result.get("checkpoint"),
                        "checkpoint_sha256": result.get("checkpoint_sha256"),
                        "sample_count": result.get("sample_count"),
                        "view_sample_counts": json.dumps(
                            result.get("view_sample_counts", {}), sort_keys=True
                        ),
                        **{metric: metrics.get(metric) for metric in CSV_METRICS},
                        "error": result.get("error"),
                    }
                )

    lines = [
        "# Stage 1 Ablation 3: cross-dataset generalization",
        "",
        f"Encoder: `{MODEL}`. Training views: `{', '.join(PCL_TYPES)}`. "
        f"Training batch size: `{batch_size}`.",
        "Rows are checkpoint training datasets. Columns are held-out test datasets. "
        "Validation selects epochs; test results never select a model.",
    ]
    for metric, title in MATRIX_METRICS:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Training \\ Test | "
                + " | ".join(DATASET_LABELS[item] for item in DATASETS)
                + " |",
                "|---|" + "---:|" * len(DATASETS),
            ]
        )
        for row_index, training_dataset in enumerate(DATASETS):
            values = [
                "-" if value is None else f"{value:.4f}"
                for value in matrices[metric][row_index]
            ]
            lines.append(
                f"| {DATASET_LABELS[training_dataset]} | " + " | ".join(values) + " |"
            )
        lines.extend(["", f"![{title}](matrix_{metric}.png)"])
        write_heatmap(output / f"matrix_{metric}.png", title, matrices[metric])
    (output / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _training_result(
    *,
    dataset: str,
    batch_size: int,
    checkpoint: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "complete",
        "model": MODEL,
        "dataset": dataset,
        "pcl_types": list(PCL_TYPES),
        "batch_size": batch_size,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "training": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    if not (args.processed_root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"processed dataset missing at {args.processed_root}; run preprocessing first"
        )
    for dataset in DATASETS:
        for split in ("train", "val", "test"):
            selected = ProcessedPlantDataset(
                args.processed_root, split=split, dataset=dataset
            )
            if not len(selected):
                raise ValueError(
                    f"Stage 1 Ablation 3 requires a non-empty {split} split for {dataset}"
                )
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Stage 1 Ablation 3 requires a CUDA GPU visible inside Docker")
    ensure_backbone_available(MODEL)

    args.output.mkdir(parents=True, exist_ok=True)
    training_results: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    progress = Ablation3Progress(
        args.output / "progress.jsonl", len(DATASETS) + len(DATASETS) ** 2
    )
    failures = 0

    for run_index, dataset in enumerate(DATASETS):
        run_output = args.output / dataset
        checkpoint = run_output / "best.ckpt"
        complete_path = run_output / "training_complete.json"
        run_output.mkdir(parents=True, exist_ok=True)
        try:
            if args.resume and complete_path.is_file():
                marker = _read_json(complete_path)
                _require_training_marker(
                    marker,
                    dataset=dataset,
                    batch_size=args.batch_size,
                    checkpoint=checkpoint,
                )
                training_results[dataset] = marker
                progress.emit(
                    run_index,
                    dataset,
                    None,
                    "train",
                    1.0,
                    "training complete (resume: skipped)",
                    status="skipped",
                )
                continue

            progress.emit(
                run_index,
                dataset,
                None,
                "train",
                0.0,
                f"train epoch 0/{args.max_epochs}",
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "tomato_recon.train.train_encoder",
                "--config",
                f"configs/encoder/{MODEL}.yaml",
                f"data.processed_root={args.processed_root}",
                f"data.dataset={dataset}",
                _pcl_types_override(PCL_TYPES),
                f"output.dir={run_output}",
                f"trainer.max_epochs={args.max_epochs}",
                f"trainer.batch_size={args.batch_size}",
                "trainer.num_workers=0",
                "seed=42",
            ]
            if args.allow_cpu:
                command.append("trainer.devices=0")
            last_checkpoint = run_output / "last.ckpt"
            if args.resume and last_checkpoint.is_file():
                _require_resume_checkpoint(
                    last_checkpoint,
                    dataset=dataset,
                    batch_size=args.batch_size,
                )
                command.extend(["--resume", str(last_checkpoint)])

            def training_event(event: dict[str, Any]) -> None:
                epoch = int(event["epoch"])
                epoch_total = int(event["epoch_total"])
                progress.emit(
                    run_index,
                    dataset,
                    None,
                    "train",
                    epoch / max(epoch_total, 1),
                    f"train epoch {epoch}/{epoch_total}",
                    epoch=epoch,
                    epoch_total=epoch_total,
                )

            _run(command, training_event)
            result = _training_result(
                dataset=dataset,
                batch_size=args.batch_size,
                checkpoint=checkpoint,
                metrics=_read_json(run_output / "metrics.json"),
            )
            complete_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            training_results[dataset] = result
            progress.emit(
                run_index,
                dataset,
                None,
                "train",
                1.0,
                "training complete",
                status="complete",
            )
        except Exception as exc:  # independent datasets should continue
            failures += 1
            result = {
                "status": "failed",
                "model": MODEL,
                "dataset": dataset,
                "pcl_types": list(PCL_TYPES),
                "batch_size": args.batch_size,
                "error": f"{type(exc).__name__}: {exc}",
            }
            training_results[dataset] = result
            (run_output / "training_failure.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            progress.emit(
                run_index,
                dataset,
                None,
                "train",
                1.0,
                f"failed: {exc}",
                status="failed",
                error=str(exc),
            )

    evaluation_index = len(DATASETS)
    for training_dataset in DATASETS:
        training_result = training_results[training_dataset]
        for evaluation_dataset in DATASETS:
            run_index = evaluation_index
            evaluation_index += 1
            result_key = f"{training_dataset}->{evaluation_dataset}"
            if training_result.get("status") != "complete":
                result = {
                    "status": "unavailable",
                    "model": MODEL,
                    "training_dataset": training_dataset,
                    "evaluation_dataset": evaluation_dataset,
                    "pcl_types": list(PCL_TYPES),
                    "batch_size": args.batch_size,
                    "error": "training checkpoint is unavailable",
                }
                evaluations[result_key] = result
                progress.emit(
                    run_index,
                    training_dataset,
                    evaluation_dataset,
                    "evaluate",
                    1.0,
                    "unavailable: training failed",
                    status="unavailable",
                )
                continue

            checkpoint = Path(str(training_result["checkpoint"]))
            checkpoint_sha = str(training_result["checkpoint_sha256"])
            evaluation_output = (
                args.output
                / training_dataset
                / "by_test_dataset"
                / evaluation_dataset
            )
            metrics_path = evaluation_output / "test_metrics.json"
            complete_path = evaluation_output / "run_complete.json"
            evaluation_output.mkdir(parents=True, exist_ok=True)
            try:
                if args.resume and complete_path.is_file():
                    result = _read_json(complete_path)
                    _require_evaluation_marker(
                        result,
                        training_dataset=training_dataset,
                        evaluation_dataset=evaluation_dataset,
                        batch_size=args.batch_size,
                        checkpoint_sha=checkpoint_sha,
                    )
                    evaluations[result_key] = result
                    progress.emit(
                        run_index,
                        training_dataset,
                        evaluation_dataset,
                        "evaluate",
                        1.0,
                        "test complete (resume: skipped)",
                        status="skipped",
                    )
                    continue

                progress.emit(
                    run_index,
                    training_dataset,
                    evaluation_dataset,
                    "evaluate",
                    0.0,
                    "evaluate held-out test split",
                )
                command = [
                    sys.executable,
                    "-u",
                    "scripts/evaluate_encoder.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--processed-root",
                    str(args.processed_root),
                    "--split",
                    "test",
                    "--dataset",
                    evaluation_dataset,
                    "--pcl-types",
                    *PCL_TYPES,
                    "--output",
                    str(metrics_path),
                    "--device",
                    "cpu" if args.allow_cpu else "cuda",
                ]
                if evaluation_dataset != training_dataset:
                    command.append("--allow-dataset-mismatch")
                _run(command)
                evaluation_report = _read_json(metrics_path)
                result = {
                    "status": "complete",
                    "model": MODEL,
                    "training_dataset": training_dataset,
                    "evaluation_dataset": evaluation_dataset,
                    "pcl_types": list(PCL_TYPES),
                    "batch_size": args.batch_size,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "sample_count": evaluation_report.get("sample_count"),
                    "view_sample_counts": evaluation_report.get(
                        "view_sample_counts", {}
                    ),
                    "metrics": evaluation_report["metrics"],
                    "metrics_path": str(metrics_path),
                }
                complete_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                evaluations[result_key] = result
                progress.emit(
                    run_index,
                    training_dataset,
                    evaluation_dataset,
                    "evaluate",
                    1.0,
                    f"test overall {result['metrics']['overall_score']:.4f}",
                    status="complete",
                    test_overall_score=result["metrics"]["overall_score"],
                )
            except Exception as exc:  # preserve every other matrix cell
                failures += 1
                result = {
                    "status": "failed",
                    "model": MODEL,
                    "training_dataset": training_dataset,
                    "evaluation_dataset": evaluation_dataset,
                    "pcl_types": list(PCL_TYPES),
                    "batch_size": args.batch_size,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                evaluations[result_key] = result
                (evaluation_output / "failure.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                progress.emit(
                    run_index,
                    training_dataset,
                    evaluation_dataset,
                    "evaluate",
                    1.0,
                    f"failed: {exc}",
                    status="failed",
                    error=str(exc),
                )

    write_ablation_3_report(
        args.output,
        batch_size=args.batch_size,
        training_results=training_results,
        evaluations=evaluations,
    )
    if failures:
        raise SystemExit(f"Stage 1 Ablation 3 completed with {failures} failed run(s)")
    print(
        f"Stage 1 Ablation 3 complete: wrote {args.output / 'matrix.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
