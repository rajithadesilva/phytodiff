from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import pytest

from scripts.evaluate_encoder import resolve_evaluation_point_cloud_types
from scripts.run_stage1_ablation_2 import (
    MATRIX_METRICS,
    PCL_TYPE_COMBINATIONS,
    _require_evaluation_marker,
    _require_training_marker,
    _sha256,
    combination_key,
    load_ablation_1_winner,
    write_ablation_2_report,
)


def test_ablation_2_uses_every_nonempty_view_combination_in_declared_order() -> None:
    expected_subsets = {
        subset
        for size in range(1, 4)
        for subset in combinations(("full", "top_down", "side"), size)
    }
    assert len(PCL_TYPE_COMBINATIONS) == 7
    assert set(PCL_TYPE_COMBINATIONS) == expected_subsets
    assert PCL_TYPE_COMBINATIONS[0] == ("full", "top_down", "side")
    assert PCL_TYPE_COMBINATIONS[-3:] == (("full",), ("top_down",), ("side",))
    assert combination_key(("full", "side")) == "full+side"


def test_cross_view_evaluation_requires_explicit_permission() -> None:
    training = ("full", "side")
    assert resolve_evaluation_point_cloud_types(
        training, None, allow_mismatch=False
    ) == training
    assert resolve_evaluation_point_cloud_types(
        training, ["full", "side"], allow_mismatch=False
    ) == training
    with pytest.raises(ValueError, match="allow-pcl-type-mismatch"):
        resolve_evaluation_point_cloud_types(
            training, ["top_down"], allow_mismatch=False
        )
    assert resolve_evaluation_point_cloud_types(
        training, ["top_down"], allow_mismatch=True
    ) == ("top_down",)


def test_ablation_1_winner_is_required_and_validated(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run stage1-ablation-1 first"):
        load_ablation_1_winner(tmp_path)

    checkpoint = tmp_path / "combined" / "kpconvx" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "winner.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "model": "kpconvx",
                "dataset": "combined",
                "pcl_types": ["full"],
                "checkpoint": str(checkpoint),
            }
        )
    )
    winner = load_ablation_1_winner(tmp_path)
    assert winner["model"] == "kpconvx"
    assert winner["dataset"] == "combined"
    assert winner["pcl_types"] == ["full"]


def test_resume_records_require_exact_views_and_checkpoint_checksum(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    checksum = _sha256(checkpoint)
    training_marker = {
        "status": "complete",
        "model": "kpconvx",
        "dataset": "combined",
        "pcl_types": ["full", "side"],
        "checkpoint_sha256": checksum,
    }
    _require_training_marker(
        training_marker,
        model="kpconvx",
        dataset="combined",
        pcl_types=("full", "side"),
        checkpoint=checkpoint,
    )
    with pytest.raises(ValueError, match="stale"):
        _require_training_marker(
            training_marker,
            model="kpconvx",
            dataset="combined",
            pcl_types=("side", "full"),
            checkpoint=checkpoint,
        )

    evaluation_marker = {
        "status": "complete",
        "model": "kpconvx",
        "dataset": "combined",
        "training_pcl_types": ["full", "side"],
        "evaluation_pcl_types": ["top_down"],
        "checkpoint_sha256": checksum,
    }
    _require_evaluation_marker(
        evaluation_marker,
        model="kpconvx",
        dataset="combined",
        training_pcl_types=("full", "side"),
        evaluation_pcl_types=("top_down",),
        checkpoint_sha256=checksum,
    )
    with pytest.raises(ValueError, match="stale"):
        _require_evaluation_marker(
            evaluation_marker,
            model="kpconvx",
            dataset="combined",
            training_pcl_types=("full", "side"),
            evaluation_pcl_types=("side",),
            checkpoint_sha256=checksum,
        )


def test_ablation_2_writes_complete_7_by_7_reports_and_task_plots(
    tmp_path: Path,
) -> None:
    winner = {
        "status": "complete",
        "model": "kpconvx",
        "dataset": "combined",
        "pcl_types": ["full"],
        "checkpoint": "outputs/stage1_ablation_1/combined/kpconvx/best.ckpt",
    }
    training_results = {}
    evaluations = {}
    for row, training_pcl_types in enumerate(PCL_TYPE_COMBINATIONS):
        training_key = combination_key(training_pcl_types)
        training_results[training_key] = {
            "status": "complete",
            "model": "kpconvx",
            "dataset": "combined",
            "pcl_types": list(training_pcl_types),
            "checkpoint": f"{training_key}/best.ckpt",
            "checkpoint_sha256": f"sha-{row}",
        }
        for column, evaluation_pcl_types in enumerate(PCL_TYPE_COMBINATIONS):
            evaluation_key = combination_key(evaluation_pcl_types)
            score = 0.4 + 0.01 * row + 0.001 * column
            evaluations[f"{training_key}->{evaluation_key}"] = {
                "status": "complete",
                "model": "kpconvx",
                "dataset": "combined",
                "training_pcl_types": list(training_pcl_types),
                "evaluation_pcl_types": list(evaluation_pcl_types),
                "checkpoint": f"{training_key}/best.ckpt",
                "checkpoint_sha256": f"sha-{row}",
                "sample_count": 5 * len(evaluation_pcl_types),
                "view_sample_counts": {
                    view: 5 for view in evaluation_pcl_types
                },
                "metrics": {
                    metric: score for metric, _ in MATRIX_METRICS
                },
            }

    write_ablation_2_report(
        tmp_path,
        winner=winner,
        training_results=training_results,
        evaluations=evaluations,
    )

    report = json.loads((tmp_path / "matrix.json").read_text())
    assert len(report["configuration_order"]) == 7
    assert len(report["evaluations"]) == 49
    assert report["test_used_for_selection"] is False
    for metric, _ in MATRIX_METRICS:
        assert len(report["matrices"][metric]) == 7
        assert all(len(row) == 7 for row in report["matrices"][metric])
        assert (tmp_path / f"matrix_{metric}.png").is_file()

    with (tmp_path / "matrix.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 49
    assert rows[0]["training_configuration"] == "full+top_down+side"
    assert rows[0]["evaluation_configuration"] == "full+top_down+side"
    assert "Rows are checkpoint training configurations" in (
        tmp_path / "matrix.md"
    ).read_text()
