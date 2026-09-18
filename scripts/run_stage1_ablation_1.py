#!/usr/bin/env python3
"""Run Stage 1 Ablation 1 across encoder architectures."""

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

from tomato_recon.data.processed import (
    POINT_CLOUD_TYPES,
    ProcessedPlantDataset,
    normalise_dataset_selection,
    normalise_point_cloud_types,
)
from tomato_recon.models.encoders.registry import ensure_backbone_available
from tomato_recon.models.pretrained import verify_sonata_checkpoint
from tomato_recon.train.common import checkpoint_sha256

MODELS = ("pointnext", "sonata_ptv3", "kpconvx")
DATASETS = ("tomatowur", "tomatopgt", "pheno4d", "combined")
SOURCE_DATASETS = tuple(dataset for dataset in DATASETS if dataset != "combined")
EVENT_PREFIX = "@@STAGE1_ABLATION_EVENT@@"


def _duration(seconds: float | None) -> str:
    if seconds is None or not torch.isfinite(torch.tensor(seconds)):
        return "--:--:--"
    seconds_int = max(int(seconds), 0)
    minutes, seconds_int = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}"


class Ablation1Progress:
    """Global, monotonic progress for the encoder-model comparison."""

    def __init__(self, output: Path, run_count: int) -> None:
        self.output = output
        self.run_count = run_count
        self.started = time.perf_counter()
        self.last_percent = 0.0
        output.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        run_index: int,
        dataset: str,
        model: str,
        phase: str,
        run_fraction: float,
        detail: str,
        *,
        status: str = "running",
        **extra: object,
    ) -> None:
        percent = 100.0 * (run_index + min(max(run_fraction, 0.0), 1.0)) / self.run_count
        percent = max(percent, self.last_percent)
        self.last_percent = percent
        elapsed = time.perf_counter() - self.started
        eta = elapsed * (100.0 - percent) / percent if percent > 0 else None
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "model": model,
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
        print(
            f"[Stage 1 Ablation 1 {percent:5.1f}%] run {run_index + 1}/{self.run_count} "
            f"{dataset}/{model} | {detail} | elapsed {_duration(elapsed)} | "
            f"ETA {_duration(eta)}",
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


def _recorded_pcl_types(values: dict[str, Any]) -> tuple[str, ...]:
    """Read ordered view metadata, including legacy single-view run records."""
    if "pcl_types" in values:
        return normalise_point_cloud_types(values["pcl_types"])
    return normalise_point_cloud_types([values.get("pcl_type", "full")])


def _single_view_visualizations_complete(
    path: Path,
    expected_ids: list[str],
    expected_view: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> bool:
    manifest_path = path / "visualization_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    if expected_view is not None and manifest.get("pcl_types") != [expected_view]:
        return False
    if (
        expected_checkpoint_sha256 is not None
        and manifest.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        return False
    renders = manifest.get("renders", {})
    return all(
        plant_id in renders
        and renders[plant_id].get("status") == "complete"
        and Path(renders[plant_id]["path"]).is_file()
        for plant_id in expected_ids
    )


def _visualizations_complete(
    path: Path,
    expected_ids: list[str],
    pcl_types: tuple[str, ...] = ("full",),
    expected_checkpoint_sha256: str | None = None,
) -> bool:
    if len(pcl_types) == 1:
        return _single_view_visualizations_complete(
            path,
            expected_ids,
            pcl_types[0],
            expected_checkpoint_sha256,
        )
    manifest_path = path / "visualization_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    if tuple(manifest.get("pcl_types", ())) != pcl_types:
        return False
    if (
        expected_checkpoint_sha256 is not None
        and manifest.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        return False
    return all(
        _single_view_visualizations_complete(
            path / view,
            expected_ids,
            view,
            expected_checkpoint_sha256,
        )
        for view in pcl_types
    )


def _require_ablation_1_record(
    record: dict[str, Any],
    *,
    dataset: str,
    model: str,
    pcl_types: tuple[str, ...],
    checkpoint: Path,
) -> None:
    expected = {
        "dataset": dataset,
        "model": model,
        "pcl_types": list(pcl_types),
    }
    actual = {
        "dataset": record.get("dataset"),
        "model": record.get("model"),
        "pcl_types": list(_recorded_pcl_types(record)),
    }
    if record.get("status") != "complete" or actual != expected:
        raise ValueError(
            f"stale Ablation 1 record at {checkpoint.parent}: "
            f"expected {expected}, got {actual}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ablation 1 checkpoint is missing: {checkpoint}")
    checksum = checkpoint_sha256(checkpoint)
    if record.get("checkpoint_sha256") != checksum:
        raise ValueError(f"Ablation 1 checkpoint checksum changed: {checkpoint}")


def _require_evaluation_report(
    report: dict[str, Any],
    *,
    dataset: str,
    pcl_types: tuple[str, ...],
    checkpoint: Path,
) -> None:
    expected = {
        "dataset": dataset,
        "pcl_types": list(pcl_types),
        "checkpoint": str(checkpoint),
    }
    actual = {
        "dataset": report.get("evaluation_dataset"),
        "pcl_types": report.get("evaluation_pcl_types", report.get("pcl_types")),
        "checkpoint": report.get("checkpoint"),
    }
    if actual != expected:
        raise ValueError(
            f"stale Ablation 1 evaluation report: expected {expected}, got {actual}"
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


def _run_key(dataset: str, model: str) -> str:
    return f"{dataset}/{model}"


def _pcl_types_override(pcl_types: tuple[str, ...]) -> str:
    """Encode an ordered view array for an OmegaConf command-line override."""
    return f"data.pcl_types=[{','.join(pcl_types)}]"


def _pcl_types_cli(pcl_types: tuple[str, ...]) -> list[str]:
    """Forward an ordered view array to an argparse ``nargs='+'`` option."""
    return ["--pcl-types", *pcl_types]


def _write_comparison(
    output: Path,
    results: dict[str, dict[str, Any]],
    combined_evaluations: dict[str, dict[str, Any]] | None = None,
    *,
    datasets: tuple[str, ...] = DATASETS,
    models: tuple[str, ...] = MODELS,
) -> None:
    combined_evaluations = combined_evaluations or {}
    dataset_reports: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        successful = {
            model: results[_run_key(dataset, model)]
            for model in models
            if results.get(_run_key(dataset, model), {}).get("status") == "complete"
        }
        ranking = sorted(
            successful,
            key=lambda model: successful[model]["validation"]["metrics"]["overall_score"],
            reverse=True,
        )
        dataset_reports[dataset] = {
            "ranking": ranking,
            "winner": ranking[0] if ranking else None,
            "models": {
                model: results.get(_run_key(dataset, model), {"status": "not_run"})
                for model in models
            },
        }

    aggregate_scores: dict[str, float] = {}
    for model in models:
        scores = [
            dataset_reports[dataset]["models"][model]["validation"]["metrics"]["overall_score"]
            for dataset in datasets
            if dataset_reports[dataset]["models"][model].get("status") == "complete"
        ]
        if len(scores) == len(datasets):
            aggregate_scores[model] = sum(scores) / len(scores)
    aggregate_ranking = sorted(
        aggregate_scores, key=aggregate_scores.__getitem__, reverse=True
    )
    all_models_complete = all(
        results.get(_run_key(dataset, model), {}).get("status") == "complete"
        for dataset in datasets
        for model in models
    )
    winner_model = aggregate_ranking[0] if all_models_complete and aggregate_ranking else None
    winner_dataset = (
        datasets[0]
        if len(datasets) == 1
        else ("combined" if "combined" in datasets else datasets[0])
    )
    winner_result = (
        results.get(_run_key(winner_dataset, winner_model), {})
        if winner_model is not None
        else {}
    )
    report = {
        "schema_version": "4.0",
        "experiment": "stage1_ablation_1",
        "pcl_types": list(
            next(
                (
                    _recorded_pcl_types(result)
                    for result in results.values()
                    if result.get("status") == "complete"
                ),
                ("full",),
            )
        ),
        "selection_split": "val",
        "test_used_for_selection": False,
        "datasets": dataset_reports,
        "aggregate_validation_mean": aggregate_scores,
        "aggregate_ranking": aggregate_ranking,
        "aggregate_winner": winner_model,
        "winner": winner_model,
        "dataset_selection": winner_dataset,
        "combined_winner": dataset_reports.get("combined", {}).get("winner"),
        "combined_models_by_dataset": {
            model: {
                dataset: combined_evaluations.get(
                    _run_key(dataset, model), {"status": "not_run"}
                )
                for dataset in SOURCE_DATASETS
                if dataset in datasets
            }
            for model in models
        },
        "runs": results,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    winner_path = output / "winner.json"
    if winner_model is not None and winner_result.get("status") == "complete":
        winner_record = {
            "schema_version": "1.0",
            "experiment": "stage1_ablation_1",
            "status": "complete",
            "selection_split": "val",
            "test_used_for_selection": False,
            "dataset": winner_dataset,
            "pcl_types": list(_recorded_pcl_types(winner_result)),
            "model": winner_model,
            "checkpoint": winner_result["checkpoint"],
            "validation_overall_score": winner_result["validation"]["metrics"][
                "overall_score"
            ],
        }
        winner_path.write_text(
            json.dumps(winner_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        winner_path.unlink(missing_ok=True)

    metric_fields = [
        "loss",
        "overall_score",
        "semantic_miou",
        "semantic_iou_background",
        "semantic_iou_leaf",
        "semantic_iou_main_stem",
        "semantic_iou_support_pole",
        "semantic_iou_side_stem",
        "skeleton_precision",
        "skeleton_recall",
        "skeleton_f1",
        "centreline_offset_mae_m",
        "centreline_offset_mae_mm",
        "centreline_offset_score",
        "junction_precision",
        "junction_recall",
        "junction_f1",
        "semantic_loss",
        "skeleton_loss",
        "offset_loss",
        "junction_loss",
    ]
    training_fields = [
        "parameter_count",
        "trainable_parameter_count",
        "best_epoch",
        "training_runtime_seconds",
        "peak_gpu_memory_mb",
    ]
    fields = [
        "dataset",
        "training_dataset",
        "evaluation_dataset",
        "model",
        "pcl_types",
        "status",
        "split",
        "sample_count",
        *metric_fields,
        *training_fields,
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in datasets:
            for model in models:
                result = results.get(_run_key(dataset, model), {"status": "not_run"})
                if result.get("status") != "complete":
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "training_dataset": dataset,
                            "evaluation_dataset": dataset,
                            "model": model,
                            "pcl_types": " ".join(_recorded_pcl_types(result)),
                            "status": result.get("status", "not_run"),
                        }
                    )
                    continue
                training = result.get("training", {})
                for split_key in ("validation", "test"):
                    metrics = result[split_key]["metrics"]
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "training_dataset": dataset,
                            "evaluation_dataset": dataset,
                            "model": model,
                            "pcl_types": " ".join(_recorded_pcl_types(result)),
                            "status": "complete",
                            "split": result[split_key]["split"],
                            "sample_count": result[split_key].get("sample_count"),
                            **{key: metrics.get(key) for key in metric_fields},
                            **{key: training.get(key) for key in training_fields},
                        }
                    )
        for model in models:
            for dataset in SOURCE_DATASETS:
                if dataset not in datasets:
                    continue
                result = combined_evaluations.get(
                    _run_key(dataset, model), {"status": "not_run"}
                )
                if result.get("status") != "complete":
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "training_dataset": "combined",
                            "evaluation_dataset": dataset,
                            "model": model,
                            "pcl_types": " ".join(_recorded_pcl_types(result)),
                            "status": f"combined_model_{result.get('status', 'not_run')}",
                        }
                    )
                    continue
                training = result.get("training", {})
                for split_key in ("validation", "test"):
                    metrics = result[split_key]["metrics"]
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "training_dataset": "combined",
                            "evaluation_dataset": dataset,
                            "model": model,
                            "pcl_types": " ".join(_recorded_pcl_types(result)),
                            "status": "complete",
                            "split": result[split_key]["split"],
                            "sample_count": result[split_key].get("sample_count"),
                            **{key: metrics.get(key) for key in metric_fields},
                            **{key: training.get(key) for key in training_fields},
                        }
                    )

    lines = [
        "# Stage 1 Ablation 1: encoder architecture",
        "",
        "Each model is ranked independently on each dataset by validation overall score. "
        "Held-out test metrics are reported only after training and are never used for selection.",
    ]
    for dataset in datasets:
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
                "| Rank | Model | Val overall | Test overall | Semantic mIoU | Skeleton F1 | Offset MAE (mm) | Junction F1 | Visualizations |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        ranking = dataset_reports[dataset]["ranking"]
        for rank, model in enumerate(ranking, start=1):
            result = dataset_reports[dataset]["models"][model]
            val = result["validation"]["metrics"]
            test = result["test"]["metrics"]
            lines.append(
                f"| {rank} | {model} | {val['overall_score']:.4f} | "
                f"{test['overall_score']:.4f} | {test['semantic_miou']:.4f} | "
                f"{test['skeleton_f1']:.4f} | {test['centreline_offset_mae_mm']:.2f} | "
                f"{test['junction_f1']:.4f} | "
                f"[{model} test set]({dataset}/{model}/test_visualizations/) |"
            )
        for model in models:
            result = dataset_reports[dataset]["models"][model]
            if result.get("status") != "complete":
                lines.append(
                    f"| - | {model} | - | - | - | - | - | - | "
                    f"{result.get('error', result.get('status', 'not run'))} |"
                )

    if len(datasets) > 1:
        lines.extend(
            [
                "",
                "## Cross-dataset validation mean",
                "",
                "This summary includes a model only after all requested dataset runs complete.",
                "",
                "| Rank | Model | Mean validation overall |",
                "|---:|---|---:|",
            ]
        )
        for rank, model in enumerate(aggregate_ranking, start=1):
            lines.append(f"| {rank} | {model} | {aggregate_scores[model]:.4f} |")

        lines.extend(
            [
                "",
                "## Combined-trained models evaluated by source dataset",
                "",
                "These checkpoints were selected using the combined validation split, then "
                "evaluated without retraining on each source subset.",
            ]
        )
    for dataset in SOURCE_DATASETS:
        if len(datasets) <= 1 or dataset not in datasets:
            continue
        lines.extend(
            [
                "",
                f"### {dataset}",
                "",
                "| Model | Val overall | Test overall | Test semantic mIoU | Test skeleton F1 | Test offset MAE (mm) | Test junction F1 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model in models:
            result = combined_evaluations.get(_run_key(dataset, model), {})
            if result.get("status") != "complete":
                lines.append(
                    f"| {model} | - | - | - | - | - | "
                    f"{result.get('error', result.get('status', 'not run'))} |"
                )
                continue
            validation = result["validation"]["metrics"]
            test = result["test"]["metrics"]
            lines.append(
                f"| {model} | {validation['overall_score']:.4f} | "
                f"{test['overall_score']:.4f} | {test['semantic_miou']:.4f} | "
                f"{test['skeleton_f1']:.4f} | "
                f"{test['centreline_offset_mae_mm']:.2f} | "
                f"{test['junction_f1']:.4f} |"
            )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    successful_runs = {
        key: result for key, result in results.items() if result.get("status") == "complete"
    }
    if successful_runs:
        overall_groups: dict[str, list[tuple[str, float]]] = {}
        for dataset in datasets:
            complete_models = dataset_reports[dataset]["models"]
            overall_groups[f"{dataset} validation"] = [
                (model, complete_models[model]["validation"]["metrics"]["overall_score"])
                for model in models
                if complete_models[model].get("status") == "complete"
            ]
            overall_groups[f"{dataset} test"] = [
                (model, complete_models[model]["test"]["metrics"]["overall_score"])
                for model in models
                if complete_models[model].get("status") == "complete"
            ]
        _chart(
            output / "overall_scores.png",
            "Stage 1 Ablation 1 overall scores",
            overall_groups,
        )
    complete_combined_evaluations = {
        key: result
        for key, result in combined_evaluations.items()
        if result.get("status") == "complete"
    }
    if complete_combined_evaluations:
        combined_groups: dict[str, list[tuple[str, float]]] = {}
        for dataset in SOURCE_DATASETS:
            if dataset not in datasets:
                continue
            combined_groups[f"{dataset} validation"] = [
                (
                    model,
                    combined_evaluations[_run_key(dataset, model)]["validation"]["metrics"][
                        "overall_score"
                    ],
                )
                for model in models
                if combined_evaluations.get(_run_key(dataset, model), {}).get("status")
                == "complete"
            ]
            combined_groups[f"{dataset} test"] = [
                (
                    model,
                    combined_evaluations[_run_key(dataset, model)]["test"]["metrics"][
                        "overall_score"
                    ],
                )
                for model in models
                if combined_evaluations.get(_run_key(dataset, model), {}).get("status")
                == "complete"
            ]
        _chart(
            output / "combined_models_by_dataset.png",
            "Combined-trained Stage 1 models evaluated by source dataset",
            combined_groups,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1_ablation_1"))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument(
        "--pcl-types",
        nargs="+",
        choices=POINT_CLOUD_TYPES,
        default=["full"],
        help="Ordered point-cloud views used for training and evaluation",
    )
    parser.add_argument(
        "--dataset",
        default="combined",
        help="One source dataset selection used for every encoder model",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Testing only")
    args = parser.parse_args()
    pcl_types = normalise_point_cloud_types(args.pcl_types)
    datasets = (normalise_dataset_selection(args.dataset),)
    models = tuple(dict.fromkeys(args.models))
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (args.output / "winner.json").unlink(missing_ok=True)

    if not (args.processed_root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"processed dataset missing at {args.processed_root}; run preprocessing first"
        )
    expected_test_ids: dict[str, list[str]] = {}
    for dataset in datasets:
        validation_dataset = ProcessedPlantDataset(
            args.processed_root, split="val", dataset=dataset
        )
        test_dataset = ProcessedPlantDataset(
            args.processed_root, split="test", dataset=dataset
        )
        if not len(validation_dataset) or not len(test_dataset):
            raise ValueError(
                f"Stage 1 Ablation 1 requires non-empty val and test splits for {dataset}"
            )
        expected_test_ids[dataset] = [
            test_dataset[index].plant_id for index in range(len(test_dataset))
        ]
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Stage 1 Ablation 1 requires a CUDA GPU visible inside Docker")
    if "sonata_ptv3" in models:
        verify_sonata_checkpoint("data/pretrained/sonata/sonata.pth")
    for model in models:
        ensure_backbone_available(model)

    runs = [(dataset, model) for dataset in datasets for model in models]
    cross_datasets: tuple[str, ...] = ()
    cross_runs = [(dataset, model) for dataset in cross_datasets for model in models]
    progress = Ablation1Progress(
        args.output / "progress.jsonl", len(runs) + len(cross_runs)
    )
    results: dict[str, dict[str, Any]] = {}
    combined_evaluations: dict[str, dict[str, Any]] = {}
    failures = 0

    for run_index, (dataset, model) in enumerate(runs):
        key = _run_key(dataset, model)
        model_output = args.output / dataset / model
        complete_path = model_output / "run_complete.json"
        if args.resume and complete_path.is_file():
            result = _read_json(complete_path)
            _require_ablation_1_record(
                result,
                dataset=dataset,
                model=model,
                pcl_types=pcl_types,
                checkpoint=model_output / "best.ckpt",
            )
            results[key] = result
            progress.emit(
                run_index,
                dataset,
                model,
                "complete",
                1.0,
                "complete (resume: skipped)",
                status="skipped",
            )
            continue
        model_output.mkdir(parents=True, exist_ok=True)
        try:
            training_marker = model_output / "training_complete.json"
            checkpoint = model_output / "best.ckpt"
            if args.resume and training_marker.is_file():
                marker = _read_json(training_marker)
                _require_ablation_1_record(
                    marker,
                    dataset=dataset,
                    model=model,
                    pcl_types=pcl_types,
                    checkpoint=checkpoint,
                )
            if not (args.resume and training_marker.is_file() and checkpoint.is_file()):
                progress.emit(
                    run_index,
                    dataset,
                    model,
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
                    f"configs/encoder/{model}.yaml",
                    f"data.processed_root={args.processed_root}",
                    f"data.dataset={dataset}",
                    _pcl_types_override(pcl_types),
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
                        run_index,
                        dataset,
                        model,
                        "train",
                        0.90 * epoch / max(epoch_total, 1),
                        f"train epoch {epoch}/{epoch_total}",
                        epoch=epoch,
                        epoch_total=epoch_total,
                    )

                _run(command, training_event)
                training_marker.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "dataset": dataset,
                            "model": model,
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": checkpoint_sha256(checkpoint),
                            "pcl_types": list(pcl_types),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                progress.emit(
                    run_index,
                    dataset,
                    model,
                    "train",
                    0.90,
                    "training complete (resume: skipped)",
                )

            validation_path = model_output / "validation_metrics.json"
            test_path = model_output / "test_metrics.json"
            for split_index, (split, destination) in enumerate(
                (("val", validation_path), ("test", test_path)), start=1
            ):
                if args.resume and destination.is_file():
                    _require_evaluation_report(
                        _read_json(destination),
                        dataset=dataset,
                        pcl_types=pcl_types,
                        checkpoint=checkpoint,
                    )
                else:
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
                            "--dataset",
                            dataset,
                            "--output",
                            str(destination),
                            "--device",
                            "cpu" if args.allow_cpu else "cuda",
                        ]
                    )
                progress.emit(
                    run_index,
                    dataset,
                    model,
                    "evaluate",
                    0.90 + 0.025 * split_index,
                    f"evaluate {split} complete",
                    split=split,
                )

            visual_output = model_output / "test_visualizations"
            best_checkpoint_sha256 = checkpoint_sha256(checkpoint)
            if not (
                args.resume
                and _visualizations_complete(
                    visual_output,
                    expected_test_ids[dataset],
                    pcl_types,
                    best_checkpoint_sha256,
                )
            ):
                progress.emit(
                    run_index,
                    dataset,
                    model,
                    "visualize",
                    0.95,
                    f"visualize test view sample 0/"
                    f"{len(expected_test_ids[dataset]) * len(pcl_types)}",
                )

                def visual_event(event: dict[str, Any]) -> None:
                    plant = int(event["plant"])
                    plant_total = int(event["plant_total"])
                    progress.emit(
                        run_index,
                        dataset,
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
                        "--dataset",
                        dataset,
                        "--count",
                        "0",
                        *_pcl_types_cli(pcl_types),
                        "--output",
                        str(visual_output),
                        "--device",
                        "cpu" if args.allow_cpu else "cuda",
                    ],
                    visual_event,
                )
            if not _visualizations_complete(
                visual_output,
                expected_test_ids[dataset],
                pcl_types,
                best_checkpoint_sha256,
            ):
                raise RuntimeError("test visualization manifest is incomplete")

            training = _read_json(model_output / "metrics.json")
            result = {
                "status": "complete",
                "dataset": dataset,
                "model": model,
                "pcl_types": list(pcl_types),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": best_checkpoint_sha256,
                "training": training,
                "validation": _read_json(validation_path),
                "test": _read_json(test_path),
                "test_visualizations": str(visual_output),
            }
            complete_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            results[key] = result
            val_score = result["validation"]["metrics"]["overall_score"]
            progress.emit(
                run_index,
                dataset,
                model,
                "complete",
                1.0,
                f"complete | val overall {val_score:.4f}",
                status="complete",
                validation_overall_score=val_score,
            )
        except Exception as exc:  # keep independent Ablation 1 runs progressing
            failures += 1
            result = {
                "status": "failed",
                "dataset": dataset,
                "model": model,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results[key] = result
            (model_output / "failure.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            progress.emit(
                run_index,
                dataset,
                model,
                "failed",
                1.0,
                f"failed: {exc}",
                status="failed",
                error=str(exc),
            )
        _write_comparison(
            args.output,
            results,
            combined_evaluations,
            datasets=datasets,
            models=models,
        )

    for cross_index, (dataset, model) in enumerate(cross_runs, start=len(runs)):
        key = _run_key(dataset, model)
        combined_result = results.get(_run_key("combined", model), {})
        evaluation_output = args.output / "combined" / model / "by_dataset" / dataset
        complete_path = evaluation_output / "run_complete.json"
        if complete_path.is_file():
            existing_evaluation = _read_json(complete_path)
            if _recorded_pcl_types(existing_evaluation) != pcl_types:
                raise ValueError(
                    f"{evaluation_output} contains point-cloud views "
                    f"{_recorded_pcl_types(existing_evaluation)!r}; choose a separate "
                    f"--output for {pcl_types!r}"
                )
            combined_evaluations[key] = existing_evaluation
            progress.emit(
                cross_index,
                f"combined->{dataset}",
                model,
                "complete",
                1.0,
                "combined-model evaluation complete (skipped)",
                status="skipped",
            )
            continue

        evaluation_output.mkdir(parents=True, exist_ok=True)
        try:
            if combined_result.get("status") != "complete":
                raise RuntimeError(
                    f"combined training result for {model} is unavailable or incomplete"
                )
            checkpoint = Path(str(combined_result["checkpoint"]))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"combined checkpoint not found: {checkpoint}")
            validation_path = evaluation_output / "validation_metrics.json"
            test_path = evaluation_output / "test_metrics.json"
            for split_index, (split, destination) in enumerate(
                (("val", validation_path), ("test", test_path)), start=1
            ):
                progress.emit(
                    cross_index,
                    f"combined->{dataset}",
                    model,
                    "evaluate",
                    0.5 * (split_index - 1),
                    f"evaluate {split}",
                    split=split,
                )
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
                        "--dataset",
                        dataset,
                        "--output",
                        str(destination),
                        "--device",
                        "cpu" if args.allow_cpu else "cuda",
                    ]
                )
            result = {
                "status": "complete",
                "training_dataset": "combined",
                "evaluation_dataset": dataset,
                "model": model,
                "pcl_types": list(pcl_types),
                "checkpoint": str(checkpoint),
                "training": combined_result.get("training", {}),
                "validation": _read_json(validation_path),
                "test": _read_json(test_path),
            }
            complete_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            combined_evaluations[key] = result
            progress.emit(
                cross_index,
                f"combined->{dataset}",
                model,
                "complete",
                1.0,
                f"complete | test overall {result['test']['metrics']['overall_score']:.4f}",
                status="complete",
                validation_overall_score=result["validation"]["metrics"]["overall_score"],
                test_overall_score=result["test"]["metrics"]["overall_score"],
            )
        except Exception as exc:
            failures += 1
            result = {
                "status": "failed",
                "training_dataset": "combined",
                "evaluation_dataset": dataset,
                "model": model,
                "error": f"{type(exc).__name__}: {exc}",
            }
            combined_evaluations[key] = result
            (evaluation_output / "failure.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            progress.emit(
                cross_index,
                f"combined->{dataset}",
                model,
                "failed",
                1.0,
                f"failed: {exc}",
                status="failed",
                error=str(exc),
            )
        _write_comparison(
            args.output,
            results,
            combined_evaluations,
            datasets=datasets,
            models=models,
        )

    _write_comparison(
        args.output,
        results,
        combined_evaluations,
        datasets=datasets,
        models=models,
    )
    if failures:
        raise SystemExit(f"Stage 1 Ablation 1 completed with {failures} failed run(s)")
    report = _read_json(args.output / "comparison.json")
    print(
        "Stage 1 Ablation 1 complete; "
        f"validation winner: {report['aggregate_winner']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
