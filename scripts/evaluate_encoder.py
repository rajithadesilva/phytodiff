#!/usr/bin/env python3
"""Evaluate a Stage 1 checkpoint on a complete processed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from tomato_recon.data.tomatowur import ProcessedTomatoDataset
from tomato_recon.evaluation.encoder import evaluate_encoder_model
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.train.common import create_split_loader, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("stage") != "encoder":
        raise ValueError(f"expected encoder checkpoint, found {payload.get('stage')!r}")
    cfg = OmegaConf.create(payload["config"])
    cfg.data.processed_root = str(args.processed_root)
    cfg.trainer.batch_size = 1
    cfg.trainer.num_workers = 0
    cfg.trainer.fast_dev_run = False
    dataset = ProcessedTomatoDataset(args.processed_root, split=args.split)
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
        expected_preprocessing_hash=first.metadata.get("preprocessing_hash"),
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
        stage=f"encoder/{args.split}",
    )
    report = {
        "schema_version": "1.0",
        "split": args.split,
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
