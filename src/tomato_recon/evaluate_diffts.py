"""Held-out stage 2 metrics, using top-down views by default."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from tomato_recon.data.diffts import from_sample, sample_points
from tomato_recon.data.processed import PointCloudViewDataset, processed_dataset_compatibility
from tomato_recon.models.diffusion.diffts_runtime import (
    create_model, create_stage1_backbone, load_stage2_checkpoint, require_stage2_schema, stage1_features,
)
from tomato_recon.models.diffusion.diffts_sampling import (
    create_scheduler, node_chamfer, sample, skeleton_chamfer,
)


def evaluate(checkpoint_path: Path, output: Path, *, split: str = "test",
             all_views: bool = False, steps: int | None = None) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require_stage2_schema(checkpoint)
    cfg = OmegaConf.create(checkpoint["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DiffTS evaluation requires a CUDA GPU")
    compatibility = processed_dataset_compatibility(cfg.data.processed_root, "combined")
    dataset = PointCloudViewDataset(
        cfg.data.processed_root, split=split, dataset="combined",
        pcl_types=("full", "top_down", "side") if all_views else ("top_down",),
    )
    if split == "test" and len(dataset.full_dataset) != 42:
        raise ValueError("expected all 42 held-out test plants")
    backbone, _ = create_stage1_backbone(cfg, device, load_source=False)
    model = create_model(cfg).to(device).eval()
    load_stage2_checkpoint(checkpoint_path, model, backbone=backbone,
                           compatibility=compatibility, checkpoint=checkpoint)
    if backbone is not None:
        backbone.eval()
    solver = create_scheduler(
        int(cfg.model.diffusion.train_timesteps),
        float(cfg.model.diffusion.beta_start),
        float(cfg.model.diffusion.beta_end),
    )
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            data = sample_points(from_sample(dataset[index]),
                                 int(cfg.model.diffusion.max_points),
                                 seed=int(cfg.seed) + index).to(device)
            point_features, global_features = stage1_features(backbone, data)
            condition = model.encode_condition(data.xyz, data.rgb, data.rgb_available,
                                               point_features, global_features)
            prediction = sample(
                model, condition, data.xyz,
                steps=steps or int(cfg.model.diffusion.sample_steps),
                guidance=float(cfg.model.diffusion.guidance),
                seed=int(cfg.seed), scheduler=solver,
            )
            row = {
                "plant_id": data.plant_id, "dataset": data.dataset, "view": data.view,
                "node_chamfer_m": float(node_chamfer(prediction.node_xyz, data.target_xyz)),
                "skeleton_chamfer_m": float(skeleton_chamfer(prediction, data.target_xyz, data.target_parent)),
                "count_absolute_error": abs(prediction.predicted_count - len(data.target_xyz)),
            }
            rows.append(row)
            print(f"{index + 1}/{len(dataset)} {data.plant_id} {data.view}: {row['skeleton_chamfer_m']:.5f} m", flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["view"])].append(row)
    def aggregate(items):
        return {key: sum(row[key] for row in items) / len(items)
                for key in ("node_chamfer_m", "skeleton_chamfer_m", "count_absolute_error")}
    report = {
        "schema_version": "diffts-stage2-evaluation-v1",
        "variant": model.variant, "split": split, "sample_count": len(rows),
        "overall": aggregate(rows),
        "by_source_view": {
            f"{source}/{view}": {"count": len(items), **aggregate(items)}
            for (source, view), items in groups.items()
        },
        "samples": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--all-views", action="store_true")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args(argv)
    report = evaluate(args.checkpoint, args.output, split=args.split,
                      all_views=args.all_views, steps=args.steps)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
