#!/usr/bin/env python3
"""Evaluate a Stage 1 checkpoint on a complete processed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from tomato_recon.data.processed import (
    POINT_CLOUD_TYPES,
    normalise_dataset_selection,
    normalise_point_cloud_types,
    processed_dataset_compatibility,
)
from tomato_recon.evaluation.encoder import evaluate_encoder_model
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.train.common import (
    create_point_cloud_dataset,
    create_split_loader,
    load_checkpoint,
    selected_point_cloud_types,
)


def resolve_evaluation_point_cloud_types(
    training_pcl_types: tuple[str, ...],
    requested_pcl_types: list[str] | None,
    *,
    allow_mismatch: bool,
) -> tuple[str, ...]:
    """Resolve evaluation views without weakening ordinary checkpoint validation."""
    evaluation_pcl_types = (
        training_pcl_types
        if requested_pcl_types is None
        else normalise_point_cloud_types(requested_pcl_types)
    )
    if evaluation_pcl_types != training_pcl_types and not allow_mismatch:
        raise ValueError(
            "evaluation point-cloud types differ from checkpoint training types: "
            f"training={training_pcl_types!r}, evaluation={evaluation_pcl_types!r}; "
            "pass --allow-pcl-type-mismatch only for a controlled cross-view evaluation"
        )
    return evaluation_pcl_types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument(
        "--dataset",
        help="Source dataset ID, or 'combined'; defaults to the checkpoint selection",
    )
    parser.add_argument(
        "--pcl-types",
        nargs="+",
        choices=POINT_CLOUD_TYPES,
        help="Ordered evaluation views; defaults to the checkpoint training views",
    )
    parser.add_argument(
        "--allow-pcl-type-mismatch",
        action="store_true",
        help="Allow evaluation views to differ from the checkpoint training views",
    )
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help=(
            "Allow controlled cross-dataset evaluation when checkpoint and target "
            "dataset contracts use the same schema and layout"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("stage") != "encoder":
        raise ValueError(f"expected encoder checkpoint, found {payload.get('stage')!r}")
    cfg = OmegaConf.create(payload["config"])
    cfg.data.processed_root = str(args.processed_root)
    training_pcl_types = selected_point_cloud_types(cfg)
    evaluation_pcl_types = resolve_evaluation_point_cloud_types(
        training_pcl_types,
        args.pcl_types,
        allow_mismatch=args.allow_pcl_type_mismatch,
    )
    training_dataset = normalise_dataset_selection(
        cfg.data.get("dataset", "combined")
    )
    dataset_selection = normalise_dataset_selection(
        args.dataset or training_dataset
    )
    cfg.data.dataset = dataset_selection
    cfg.data.pcl_types = list(evaluation_pcl_types)
    cfg.trainer.batch_size = 1
    cfg.trainer.num_workers = 0
    cfg.trainer.fast_dev_run = False
    dataset = create_point_cloud_dataset(cfg, args.split)
    if not len(dataset):
        raise ValueError(f"split {args.split!r} is empty at {args.processed_root}")
    first = dataset[0]
    model = PointEncoder(
        create_backbone_from_config(cfg.model.encoder),
        int(cfg.model.encoder.num_semantic_classes),
    )
    checkpoint = load_checkpoint(
        args.checkpoint,
        model,
        expected_dataset_compatibility=processed_dataset_compatibility(
            args.processed_root, dataset_selection
        ),
        expected_pcl_types=training_pcl_types,
        allow_dataset_subset=True,
        allow_dataset_mismatch=args.allow_dataset_mismatch,
        expected_max_nodes=len(first.node_xyz),
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested, but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    model.to(device)
    metrics = evaluate_encoder_model(
        model,
        create_split_loader(cfg, args.split, shuffle=False),
        device,
        num_classes=int(cfg.model.encoder.num_semantic_classes),
        skeleton_threshold_m=float(cfg.model.encoder.skeleton_threshold_m),
        junction_threshold_multiplier=float(
            cfg.model.encoder.junction_threshold_multiplier
        ),
        stage=f"encoder/{args.split}",
    )
    report = {
        "schema_version": "1.0",
        "split": args.split,
        "training_dataset": training_dataset,
        "training_pcl_types": list(training_pcl_types),
        "dataset": dataset_selection,
        "evaluation_dataset": dataset_selection,
        "pcl_types": list(evaluation_pcl_types),
        "evaluation_pcl_types": list(evaluation_pcl_types),
        "dataset_mismatch_allowed": bool(args.allow_dataset_mismatch),
        "view_sample_counts": {
            view: len(dataset.full_dataset) for view in evaluation_pcl_types
        },
        "sample_count": len(dataset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{args.split} overall={metrics['overall_score']:.4f}; wrote {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
