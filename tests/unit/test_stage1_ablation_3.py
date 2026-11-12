from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from scripts.run_stage1_ablation_3 import (
    DATASETS,
    MATRIX_METRICS,
    MODEL,
    PCL_TYPES,
    _positive_int,
    _require_evaluation_marker,
    _require_resume_checkpoint,
    _require_training_marker,
    build_parser,
    write_ablation_3_report,
)
from tomato_recon.train.common import checkpoint_sha256


def test_ablation_3_configuration_and_batch_size_contract() -> None:
    assert DATASETS == ("tomatowur", "tomatopgt", "pheno4d", "combined")
    assert MODEL == "kpconvx"
    assert PCL_TYPES == ("full", "top_down", "side")
    args = build_parser().parse_args([])
    assert args.batch_size == 1
    assert build_parser().parse_args(["--batch-size", "4"]).batch_size == 4
    with pytest.raises(Exception, match="positive integer"):
        _positive_int("0")


def test_ablation_3_resume_records_require_batch_and_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    checksum = checkpoint_sha256(checkpoint)
    training_marker = {
        "status": "complete",
        "model": MODEL,
        "dataset": "tomatowur",
        "pcl_types": list(PCL_TYPES),
        "batch_size": 2,
        "checkpoint_sha256": checksum,
    }
    _require_training_marker(
        training_marker,
        dataset="tomatowur",
        batch_size=2,
        checkpoint=checkpoint,
    )
    with pytest.raises(ValueError, match="stale"):
        _require_training_marker(
            training_marker,
            dataset="tomatowur",
            batch_size=1,
            checkpoint=checkpoint,
        )

    evaluation_marker = {
        "status": "complete",
        "model": MODEL,
        "training_dataset": "tomatowur",
        "evaluation_dataset": "pheno4d",
        "pcl_types": list(PCL_TYPES),
        "batch_size": 2,
        "checkpoint_sha256": checksum,
    }
    _require_evaluation_marker(
        evaluation_marker,
        training_dataset="tomatowur",
        evaluation_dataset="pheno4d",
        batch_size=2,
        checkpoint_sha=checksum,
    )
    with pytest.raises(ValueError, match="stale"):
        _require_evaluation_marker(
            evaluation_marker,
            training_dataset="tomatowur",
            evaluation_dataset="pheno4d",
            batch_size=1,
            checkpoint_sha=checksum,
        )


def test_ablation_3_partial_resume_checkpoint_requires_exact_configuration(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    payload = {
        "stage": "encoder",
        "config": {
            "model": {"encoder": {"name": MODEL}},
            "data": {"dataset": "tomatowur", "pcl_types": list(PCL_TYPES)},
            "trainer": {"batch_size": 2},
        },
    }
    torch.save(payload, checkpoint)
    _require_resume_checkpoint(checkpoint, dataset="tomatowur", batch_size=2)
    with pytest.raises(ValueError, match="stale"):
        _require_resume_checkpoint(checkpoint, dataset="tomatowur", batch_size=1)
    with pytest.raises(ValueError, match="stale"):
        _require_resume_checkpoint(checkpoint, dataset="pheno4d", batch_size=2)


def test_ablation_3_writes_complete_4_by_4_reports_and_heatmaps(
    tmp_path: Path,
) -> None:
    training_results = {}
    evaluations = {}
    for row, training_dataset in enumerate(DATASETS):
        training_results[training_dataset] = {
            "status": "complete",
            "model": MODEL,
            "dataset": training_dataset,
            "pcl_types": list(PCL_TYPES),
            "batch_size": 1,
            "checkpoint": f"{training_dataset}/best.ckpt",
            "checkpoint_sha256": f"sha-{row}",
        }
        for column, evaluation_dataset in enumerate(DATASETS):
            score = 0.4 + 0.01 * row + 0.001 * column
            evaluations[f"{training_dataset}->{evaluation_dataset}"] = {
                "status": "complete",
                "model": MODEL,
                "training_dataset": training_dataset,
                "evaluation_dataset": evaluation_dataset,
                "pcl_types": list(PCL_TYPES),
                "batch_size": 1,
                "checkpoint": f"{training_dataset}/best.ckpt",
                "checkpoint_sha256": f"sha-{row}",
                "sample_count": 15,
                "view_sample_counts": {view: 5 for view in PCL_TYPES},
                "metrics": {metric: score for metric, _ in MATRIX_METRICS},
            }

    write_ablation_3_report(
        tmp_path,
        batch_size=1,
        training_results=training_results,
        evaluations=evaluations,
    )

    report = json.loads((tmp_path / "matrix.json").read_text())
    assert report["dataset_order"] == list(DATASETS)
    assert report["model"] == MODEL
    assert report["pcl_types"] == list(PCL_TYPES)
    assert report["batch_size"] == 1
    assert report["test_used_for_selection"] is False
    assert len(report["training_runs"]) == 4
    assert len(report["evaluations"]) == 16
    for metric, _ in MATRIX_METRICS:
        assert len(report["matrices"][metric]) == 4
        assert all(len(row) == 4 for row in report["matrices"][metric])
        assert (tmp_path / f"matrix_{metric}.png").is_file()

    with (tmp_path / "matrix.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 16
    assert rows[0]["training_dataset"] == "tomatowur"
    assert rows[0]["evaluation_dataset"] == "tomatowur"
    assert rows[-1]["training_dataset"] == "combined"
    assert rows[-1]["evaluation_dataset"] == "combined"
    assert "Rows are checkpoint training datasets" in (
        tmp_path / "matrix.md"
    ).read_text()
