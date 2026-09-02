from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.run_stage1_benchmark import (
    DATASETS,
    MODELS,
    BenchmarkProgress,
    _write_comparison,
)
from tomato_recon.evaluation.encoder import EncoderMetricAccumulator
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.kpconvx import KPConvXAdapter
from tomato_recon.models.pretrained import verify_sonata_checkpoint


def test_kpconvx_dense_shapes_masks_and_backward() -> None:
    backbone = KPConvXAdapter(
        output_dim=16,
        global_dim=24,
        layer_blocks=(1, 1),
        neighbor_limits=(4, 4),
        init_channels=16,
        attention_groups=4,
    )
    model = PointEncoder(backbone, num_semantic_classes=5)
    xyz = torch.rand(2, 48, 3) * 0.1
    features = torch.rand(2, 48, 6)
    mask = torch.ones(2, 48, dtype=torch.bool)
    mask[1, 31:] = False
    output = model(xyz, features, mask)
    assert output.point_features.shape == (2, 48, 16)
    assert output.global_feature.shape == (2, 24)
    assert torch.count_nonzero(output.point_features[1, 31:]) == 0
    (output.semantic_logits.sum() + output.centreline_offset.sum()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_missing_sonata_checkpoint_has_preparation_hint() -> None:
    with TemporaryDirectory() as directory:
        missing = Path(directory) / "sonata.pth"
        try:
            verify_sonata_checkpoint(missing)
        except FileNotFoundError as error:
            assert "make prepare-stage1-models" in str(error)
        else:
            raise AssertionError("missing Sonata checkpoint was accepted")


def test_overall_metric_formula() -> None:
    accumulator = EncoderMetricAccumulator(1)
    accumulator.semantic_intersection[0] = 8
    accumulator.semantic_union[0] = 10
    accumulator.skeleton_tp = 8
    accumulator.skeleton_predicted = 10
    accumulator.skeleton_target = 10
    accumulator.junction_tp = 6
    accumulator.junction_predicted = 10
    accumulator.junction_target = 10
    accumulator.offset_absolute_error = 0.006
    accumulator.offset_elements = 1
    metrics = accumulator.compute()
    expected = 0.20 * 0.8 + 0.45 * 0.8 + 0.25 * math.exp(-1.0) + 0.10 * 0.6
    assert math.isclose(metrics["overall_score"], expected)


def test_benchmark_progress_is_monotonic_and_persisted() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "progress.jsonl"
        progress = BenchmarkProgress(path, run_count=12)
        progress.emit(0, "tomatowur", "pointnext", "train", 0.45, "epoch 25/50", epoch=25)
        progress.emit(0, "tomatowur", "pointnext", "train", 0.90, "epoch 50/50", epoch=50)
        progress.emit(
            0, "tomatowur", "pointnext", "complete", 1.0, "complete", status="complete"
        )
        events = [json.loads(line) for line in path.read_text().splitlines()]
        percentages = [event["overall_percent"] for event in events]
        assert percentages == sorted(percentages)
        assert math.isclose(percentages[-1], 100.0 / 12.0)
        assert all("eta_seconds" in event and "timestamp" in event for event in events)
        assert all(event["dataset"] == "tomatowur" for event in events)


def test_comparison_outputs_rank_by_validation_and_link_visualizations() -> None:
    def metrics(score: float) -> dict[str, float]:
        return {
            "overall_score": score,
            "semantic_miou": score,
            "skeleton_precision": score,
            "skeleton_recall": score,
            "skeleton_f1": score,
            "centreline_offset_mae_mm": 1.0,
            "centreline_offset_score": score,
            "junction_precision": score,
            "junction_recall": score,
            "junction_f1": score,
        }

    with TemporaryDirectory() as directory:
        output = Path(directory)
        results = {}
        combined_evaluations = {}
        for dataset in DATASETS:
            for index, model in enumerate(MODELS):
                results[f"{dataset}/{model}"] = {
                    "status": "complete",
                    "dataset": dataset,
                    "model": model,
                    "training": {"best_epoch": index + 1},
                    "validation": {
                        "split": "val",
                        "metrics": metrics(0.5 + index * 0.1),
                    },
                    "test": {
                        "split": "test",
                        "metrics": metrics(0.9 - index * 0.1),
                    },
                }
                if dataset != "combined":
                    combined_evaluations[f"{dataset}/{model}"] = {
                        "status": "complete",
                        "training_dataset": "combined",
                        "evaluation_dataset": dataset,
                        "model": model,
                        "training": {"best_epoch": index + 1},
                        "validation": {
                            "split": "val",
                            "sample_count": 2,
                            "metrics": metrics(0.45 + index * 0.1),
                        },
                        "test": {
                            "split": "test",
                            "sample_count": 3,
                            "metrics": metrics(0.75 - index * 0.1),
                        },
                    }
        _write_comparison(output, results, combined_evaluations)
        report = json.loads((output / "comparison.json").read_text())
        assert report["combined_winner"] == "kpconvx"
        assert report["aggregate_winner"] == "kpconvx"
        assert report["datasets"]["pheno4d"]["winner"] == "kpconvx"
        assert (
            report["combined_models_by_dataset"]["kpconvx"]["pheno4d"]["status"]
            == "complete"
        )
        assert report["test_used_for_selection"] is False
        assert "combined/kpconvx/test_visualizations/" in (
            output / "comparison.md"
        ).read_text()
        comparison_csv = (output / "comparison.csv").read_text()
        assert "training_dataset,evaluation_dataset" in comparison_csv
        assert "combined,pheno4d" in comparison_csv
        for name in (
            "comparison.csv",
            "overall_scores.png",
            "combined_models_by_dataset.png",
        ):
            assert (output / name).is_file()
