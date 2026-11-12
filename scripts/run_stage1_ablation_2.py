#!/usr/bin/env python3
"""Run Stage 1 Ablation 2 across every non-empty point-cloud view combination."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    from scripts.run_stage1_ablation_1 import (
        _pcl_types_override,
        _read_json,
        _recorded_batch_size,
        _require_encoder_resume_checkpoint,
        _run,
    )
else:
    from run_stage1_ablation_1 import (
        _pcl_types_override,
        _read_json,
        _recorded_batch_size,
        _require_encoder_resume_checkpoint,
        _run,
    )
from tomato_recon.data.processed import (
    ProcessedPlantDataset,
    normalise_dataset_selection,
    normalise_point_cloud_types,
)
from tomato_recon.models.encoders.registry import ensure_backbone_available
from tomato_recon.models.pretrained import verify_sonata_checkpoint

PCL_TYPE_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("full", "top_down", "side"),
    ("full", "top_down"),
    ("full", "side"),
    ("top_down", "side"),
    ("full",),
    ("top_down",),
    ("side",),
)

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


def combination_key(pcl_types: tuple[str, ...]) -> str:
    """Return the stable directory and report key for an ordered view array."""
    return "+".join(normalise_point_cloud_types(pcl_types))


def _combination_label(pcl_types: tuple[str, ...]) -> str:
    return " + ".join(pcl_types)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    value = max(int(seconds), 0)
    minutes, seconds_value = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"


class Ablation2Progress:
    """Durable progress across seven training runs and 49 evaluations."""

    def __init__(self, output: Path, run_count: int) -> None:
        self.output = output
        self.run_count = run_count
        self.started = time.perf_counter()
        self.last_percent = 0.0
        output.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        run_index: int,
        training_key: str,
        evaluation_key: str | None,
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
            "training_configuration": training_key,
            "evaluation_configuration": evaluation_key,
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
        pair = training_key if evaluation_key is None else f"{training_key}->{evaluation_key}"
        print(
            f"[Stage 1 Ablation 2 {percent:5.1f}%] run {run_index + 1}/"
            f"{self.run_count} {pair} | {detail} | elapsed {_duration(elapsed)} | "
            f"ETA {_duration(eta)}",
            flush=True,
        )


def load_ablation_1_winner(output: Path) -> dict[str, Any]:
    """Load and validate the architecture selected by Ablation 1."""
    winner_path = output / "winner.json"
    if not winner_path.is_file():
        raise FileNotFoundError(
            f"Stage 1 Ablation 1 winner is missing: {winner_path}; "
            "run stage1-ablation-1 first"
        )
    winner = _read_json(winner_path)
    if winner.get("status") != "complete":
        raise ValueError(f"Ablation 1 winner record is incomplete: {winner_path}")
    model = str(winner.get("model", "")).strip()
    if not model:
        raise ValueError(f"Ablation 1 winner record has no model: {winner_path}")
    dataset = normalise_dataset_selection(winner.get("dataset"))
    pcl_types = normalise_point_cloud_types(winner.get("pcl_types", ()))
    checkpoint_value = str(winner.get("checkpoint", "")).strip()
    if not checkpoint_value:
        raise ValueError(f"Ablation 1 winner record has no checkpoint: {winner_path}")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ablation 1 winning checkpoint is missing: {checkpoint}")
    return {
        **winner,
        "model": model,
        "dataset": dataset,
        "pcl_types": list(pcl_types),
        "batch_size": _recorded_batch_size(winner),
        "winner_record": str(winner_path),
    }


def _require_training_marker(
    marker: dict[str, Any],
    *,
    model: str,
    dataset: str,
    pcl_types: tuple[str, ...],
    checkpoint: Path,
    batch_size: int = 1,
) -> None:
    expected = {
        "model": model,
        "dataset": dataset,
        "pcl_types": list(pcl_types),
        "batch_size": batch_size,
    }
    actual = {
        **{key: marker.get(key) for key in expected if key != "batch_size"},
        "batch_size": _recorded_batch_size(marker),
    }
    if marker.get("status") != "complete" or actual != expected:
        raise ValueError(
            f"stale Ablation 2 training record at {checkpoint.parent}: "
            f"expected {expected}, got {actual}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ablation 2 checkpoint is missing: {checkpoint}")
    actual_checksum = _sha256(checkpoint)
    if marker.get("checkpoint_sha256") != actual_checksum:
        raise ValueError(f"Ablation 2 checkpoint checksum changed: {checkpoint}")


def _require_evaluation_marker(
    marker: dict[str, Any],
    *,
    model: str,
    dataset: str,
    training_pcl_types: tuple[str, ...],
    evaluation_pcl_types: tuple[str, ...],
    checkpoint_sha256: str,
    batch_size: int = 1,
) -> None:
    expected = {
        "model": model,
        "dataset": dataset,
        "training_pcl_types": list(training_pcl_types),
        "evaluation_pcl_types": list(evaluation_pcl_types),
        "checkpoint_sha256": checkpoint_sha256,
        "batch_size": batch_size,
    }
    actual = {
        **{key: marker.get(key) for key in expected if key != "batch_size"},
        "batch_size": _recorded_batch_size(marker),
    }
    if marker.get("status") != "complete" or actual != expected:
        raise ValueError(
            "stale Ablation 2 evaluation record: "
            f"expected {expected}, got {actual}"
        )


def _metric_matrix(
    evaluations: dict[str, dict[str, Any]], metric: str
) -> list[list[float | None]]:
    matrix: list[list[float | None]] = []
    for training_pcl_types in PCL_TYPE_COMBINATIONS:
        row: list[float | None] = []
        training_key = combination_key(training_pcl_types)
        for evaluation_pcl_types in PCL_TYPE_COMBINATIONS:
            evaluation_key = combination_key(evaluation_pcl_types)
            result = evaluations.get(f"{training_key}->{evaluation_key}", {})
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


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    top_y: float,
    lines: list[str],
    *,
    fill: tuple[int, int, int],
) -> None:
    for index, line in enumerate(lines):
        box = draw.textbbox((0, 0), line)
        width = box[2] - box[0]
        draw.text((center_x - width / 2, top_y + index * 14), line, fill=fill)


def write_heatmap(
    path: Path,
    title: str,
    matrix: list[list[float | None]],
) -> None:
    """Write a dependency-light 7x7 metric heatmap with explicit axes."""
    cell_width = 118
    cell_height = 58
    left = 205
    top = 125
    width = left + cell_width * len(PCL_TYPE_COMBINATIONS) + 25
    height = top + cell_height * len(PCL_TYPE_COMBINATIONS) + 65
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((18, 15), title, fill=(20, 20, 20))
    draw.text((18, 36), "Rows: checkpoint training views", fill=(55, 55, 55))
    draw.text((18, 52), "Columns: held-out test views", fill=(55, 55, 55))

    for column, pcl_types in enumerate(PCL_TYPE_COMBINATIONS):
        center_x = left + column * cell_width + cell_width / 2
        _draw_centered_lines(
            draw,
            center_x,
            74,
            list(pcl_types),
            fill=(40, 40, 40),
        )
    for row, training_pcl_types in enumerate(PCL_TYPE_COMBINATIONS):
        row_y = top + row * cell_height
        label = _combination_label(training_pcl_types)
        draw.text((12, row_y + 21), label, fill=(40, 40, 40))
        for column, value in enumerate(matrix[row]):
            x0 = left + column * cell_width
            y0 = row_y
            draw.rectangle(
                (x0 + 1, y0 + 1, x0 + cell_width - 2, y0 + cell_height - 2),
                fill=_heatmap_colour(value),
                outline=(245, 245, 245),
            )
            text_value = "failed" if value is None else f"{value:.4f}"
            text_box = draw.textbbox((0, 0), text_value)
            text_width = text_box[2] - text_box[0]
            draw.text(
                (x0 + (cell_width - text_width) / 2, y0 + 21),
                text_value,
                fill=(25, 25, 25),
            )
    draw.text((left, height - 35), "grey = unavailable", fill=(70, 70, 70))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_ablation_2_report(
    output: Path,
    *,
    winner: dict[str, Any],
    training_results: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
    batch_size: int = 1,
) -> None:
    """Write complete machine-readable results and one heatmap per Stage 1 task."""
    matrices = {
        metric: _metric_matrix(evaluations, metric) for metric, _ in MATRIX_METRICS
    }
    report = {
        "schema_version": "1.0",
        "experiment": "stage1_ablation_2",
        "model": winner["model"],
        "dataset": winner["dataset"],
        "batch_size": batch_size,
        "ablation_1_winner": winner,
        "selection_split": "val",
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "configuration_order": [
            {
                "key": combination_key(pcl_types),
                "pcl_types": list(pcl_types),
            }
            for pcl_types in PCL_TYPE_COMBINATIONS
        ],
        "training_runs": training_results,
        "evaluations": evaluations,
        "matrices": matrices,
    }
    (output / "matrix.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    csv_fields = [
        "training_configuration",
        "training_pcl_types",
        "evaluation_configuration",
        "evaluation_pcl_types",
        "model",
        "dataset",
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
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for training_pcl_types in PCL_TYPE_COMBINATIONS:
            training_key = combination_key(training_pcl_types)
            for evaluation_pcl_types in PCL_TYPE_COMBINATIONS:
                evaluation_key = combination_key(evaluation_pcl_types)
                result = evaluations.get(
                    f"{training_key}->{evaluation_key}",
                    {"status": "not_run"},
                )
                metrics = result.get("metrics", {})
                writer.writerow(
                    {
                        "training_configuration": training_key,
                        "training_pcl_types": " ".join(training_pcl_types),
                        "evaluation_configuration": evaluation_key,
                        "evaluation_pcl_types": " ".join(evaluation_pcl_types),
                        "model": winner["model"],
                        "dataset": winner["dataset"],
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
        "# Stage 1 Ablation 2: point-cloud configurations",
        "",
        f"Winning architecture from Ablation 1: `{winner['model']}`.",
        f"Source dataset selection: `{winner['dataset']}`.",
        f"Training batch size: `{batch_size}`.",
        "Rows are checkpoint training configurations. Columns are held-out test "
        "configurations. Validation selects epochs; test results never select a model.",
    ]
    for metric, title in MATRIX_METRICS:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Training \\ Test | "
                + " | ".join(_combination_label(item) for item in PCL_TYPE_COMBINATIONS)
                + " |",
                "|---|" + "---:|" * len(PCL_TYPE_COMBINATIONS),
            ]
        )
        for row_index, training_pcl_types in enumerate(PCL_TYPE_COMBINATIONS):
            values = [
                "-" if value is None else f"{value:.4f}"
                for value in matrices[metric][row_index]
            ]
            lines.append(
                f"| {_combination_label(training_pcl_types)} | "
                + " | ".join(values)
                + " |"
            )
        lines.extend(["", f"![{title}](matrix_{metric}.png)"])
        write_heatmap(output / f"matrix_{metric}.png", title, matrices[metric])
    (output / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _training_result(
    *,
    model: str,
    dataset: str,
    pcl_types: tuple[str, ...],
    checkpoint: Path,
    metrics: dict[str, Any],
    batch_size: int = 1,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "model": model,
        "dataset": dataset,
        "pcl_types": list(pcl_types),
        "batch_size": batch_size,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("data/dataset"))
    parser.add_argument(
        "--ablation-1-output",
        type=Path,
        default=Path("outputs/stage1_ablation_1"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1_ablation_2"))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Testing only")
    args = parser.parse_args()

    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not (args.processed_root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"processed dataset missing at {args.processed_root}; run preprocessing first"
        )
    winner = load_ablation_1_winner(args.ablation_1_output)
    if int(winner["batch_size"]) != args.batch_size:
        raise ValueError(
            "Ablation 2 batch size must match its Ablation 1 winner: "
            f"winner={winner['batch_size']}, requested={args.batch_size}; "
            "rerun Ablation 1 with the shared batch size"
        )
    model = str(winner["model"])
    dataset = str(winner["dataset"])
    validation_dataset = ProcessedPlantDataset(
        args.processed_root, split="val", dataset=dataset
    )
    test_dataset = ProcessedPlantDataset(
        args.processed_root, split="test", dataset=dataset
    )
    if not len(validation_dataset) or not len(test_dataset):
        raise ValueError(
            f"Stage 1 Ablation 2 requires non-empty val and test splits for {dataset}"
        )
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Stage 1 Ablation 2 requires a CUDA GPU visible inside Docker")
    ensure_backbone_available(model)
    if model == "sonata_ptv3":
        verify_sonata_checkpoint("data/pretrained/sonata/sonata.pth")

    args.output.mkdir(parents=True, exist_ok=True)
    training_results: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    total_runs = len(PCL_TYPE_COMBINATIONS) + len(PCL_TYPE_COMBINATIONS) ** 2
    progress = Ablation2Progress(args.output / "progress.jsonl", total_runs)
    failures = 0

    for run_index, pcl_types in enumerate(PCL_TYPE_COMBINATIONS):
        training_key = combination_key(pcl_types)
        run_output = args.output / training_key
        checkpoint = run_output / "best.ckpt"
        complete_path = run_output / "training_complete.json"
        run_output.mkdir(parents=True, exist_ok=True)
        try:
            if args.resume and complete_path.is_file():
                marker = _read_json(complete_path)
                _require_training_marker(
                    marker,
                    model=model,
                    dataset=dataset,
                    pcl_types=pcl_types,
                    checkpoint=checkpoint,
                    batch_size=args.batch_size,
                )
                training_results[training_key] = marker
                progress.emit(
                    run_index,
                    training_key,
                    None,
                    "train",
                    1.0,
                    "training complete (resume: skipped)",
                    status="skipped",
                )
                continue

            progress.emit(
                run_index,
                training_key,
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
                f"configs/encoder/{model}.yaml",
                f"data.processed_root={args.processed_root}",
                f"data.dataset={dataset}",
                _pcl_types_override(pcl_types),
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
                _require_encoder_resume_checkpoint(
                    last_checkpoint,
                    experiment="Ablation 2",
                    model=model,
                    dataset=dataset,
                    pcl_types=pcl_types,
                    batch_size=args.batch_size,
                )
                command.extend(["--resume", str(last_checkpoint)])

            def training_event(event: dict[str, Any]) -> None:
                epoch = int(event["epoch"])
                epoch_total = int(event["epoch_total"])
                progress.emit(
                    run_index,
                    training_key,
                    None,
                    "train",
                    epoch / max(epoch_total, 1),
                    f"train epoch {epoch}/{epoch_total}",
                    epoch=epoch,
                    epoch_total=epoch_total,
                )

            _run(command, training_event)
            result = _training_result(
                model=model,
                dataset=dataset,
                pcl_types=pcl_types,
                checkpoint=checkpoint,
                metrics=_read_json(run_output / "metrics.json"),
                batch_size=args.batch_size,
            )
            complete_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            training_results[training_key] = result
            progress.emit(
                run_index,
                training_key,
                None,
                "train",
                1.0,
                "training complete",
                status="complete",
            )
        except Exception as exc:  # independent configurations should continue
            failures += 1
            result = {
                "status": "failed",
                "model": model,
                "dataset": dataset,
                "pcl_types": list(pcl_types),
                "batch_size": args.batch_size,
                "error": f"{type(exc).__name__}: {exc}",
            }
            training_results[training_key] = result
            (run_output / "training_failure.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            progress.emit(
                run_index,
                training_key,
                None,
                "train",
                1.0,
                f"failed: {exc}",
                status="failed",
                error=str(exc),
            )

    evaluation_index = len(PCL_TYPE_COMBINATIONS)
    for training_pcl_types in PCL_TYPE_COMBINATIONS:
        training_key = combination_key(training_pcl_types)
        training_result = training_results[training_key]
        for evaluation_pcl_types in PCL_TYPE_COMBINATIONS:
            evaluation_key = combination_key(evaluation_pcl_types)
            result_key = f"{training_key}->{evaluation_key}"
            run_index = evaluation_index
            evaluation_index += 1
            if training_result.get("status") != "complete":
                result = {
                    "status": "unavailable",
                    "model": model,
                    "dataset": dataset,
                    "training_pcl_types": list(training_pcl_types),
                    "evaluation_pcl_types": list(evaluation_pcl_types),
                    "batch_size": args.batch_size,
                    "error": "training checkpoint is unavailable",
                }
                evaluations[result_key] = result
                progress.emit(
                    run_index,
                    training_key,
                    evaluation_key,
                    "evaluate",
                    1.0,
                    "unavailable: training failed",
                    status="unavailable",
                )
                continue

            checkpoint = Path(str(training_result["checkpoint"]))
            checkpoint_sha256 = str(training_result["checkpoint_sha256"])
            evaluation_output = (
                args.output
                / training_key
                / "by_test_pcl_types"
                / evaluation_key
            )
            metrics_path = evaluation_output / "test_metrics.json"
            complete_path = evaluation_output / "run_complete.json"
            evaluation_output.mkdir(parents=True, exist_ok=True)
            try:
                if args.resume and complete_path.is_file():
                    result = _read_json(complete_path)
                    _require_evaluation_marker(
                        result,
                        model=model,
                        dataset=dataset,
                        training_pcl_types=training_pcl_types,
                        evaluation_pcl_types=evaluation_pcl_types,
                        checkpoint_sha256=checkpoint_sha256,
                        batch_size=args.batch_size,
                    )
                    evaluations[result_key] = result
                    progress.emit(
                        run_index,
                        training_key,
                        evaluation_key,
                        "evaluate",
                        1.0,
                        "test complete (resume: skipped)",
                        status="skipped",
                    )
                    continue

                progress.emit(
                    run_index,
                    training_key,
                    evaluation_key,
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
                    dataset,
                    "--pcl-types",
                    *evaluation_pcl_types,
                    "--output",
                    str(metrics_path),
                    "--device",
                    "cpu" if args.allow_cpu else "cuda",
                ]
                if evaluation_pcl_types != training_pcl_types:
                    command.append("--allow-pcl-type-mismatch")
                _run(command)
                evaluation_report = _read_json(metrics_path)
                result = {
                    "status": "complete",
                    "model": model,
                    "dataset": dataset,
                    "training_pcl_types": list(training_pcl_types),
                    "evaluation_pcl_types": list(evaluation_pcl_types),
                    "batch_size": args.batch_size,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
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
                    training_key,
                    evaluation_key,
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
                    "model": model,
                    "dataset": dataset,
                    "training_pcl_types": list(training_pcl_types),
                    "evaluation_pcl_types": list(evaluation_pcl_types),
                    "batch_size": args.batch_size,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                evaluations[result_key] = result
                (evaluation_output / "failure.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                progress.emit(
                    run_index,
                    training_key,
                    evaluation_key,
                    "evaluate",
                    1.0,
                    f"failed: {exc}",
                    status="failed",
                    error=str(exc),
                )

    write_ablation_2_report(
        args.output,
        winner=winner,
        training_results=training_results,
        evaluations=evaluations,
        batch_size=args.batch_size,
    )
    if failures:
        raise SystemExit(f"Stage 1 Ablation 2 completed with {failures} failed run(s)")
    print(
        f"Stage 1 Ablation 2 complete: {model} on {dataset}; "
        f"wrote {args.output / 'matrix.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
