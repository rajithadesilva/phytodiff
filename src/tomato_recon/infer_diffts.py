"""Standalone target-free DiffTS inference from a saved top-down view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from tomato_recon.data.diffts import load_top_down_input, sample_points
from tomato_recon.export.ply import write_point_ply
from tomato_recon.models.diffusion.diffts_runtime import (
    create_model, load_frozen_stage1, load_stage2_checkpoint, stage1_features,
)
from tomato_recon.models.diffusion.diffts_sampling import create_scheduler, sample


def run(checkpoint_path: Path, input_path: Path, output: Path, *,
        seed: int | None = None, steps: int | None = None) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "diffts-stage2-v1":
        raise ValueError("inference requires a DiffTS stage 2 checkpoint")
    cfg = OmegaConf.create(checkpoint["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DiffTS sparse inference requires a CUDA GPU")
    data = load_top_down_input(input_path)
    with np.load(input_path, allow_pickle=False) as view:
        metadata = json.loads(str(view["metadata_json"].item()))
    allowed_hashes = checkpoint.get("dataset_compatibility", {}).get("datasets", {}).get(data.dataset, [])
    if metadata.get("preprocessing_hash") not in allowed_hashes:
        raise ValueError("top-down input preprocessing is incompatible with DiffTS checkpoint")
    data = sample_points(data, int(cfg.model.diffusion.max_points),
                         seed=int(cfg.seed)).to(device)
    encoder, stage1_hash = load_frozen_stage1(cfg, device)
    model = create_model(cfg).to(device).eval()
    load_stage2_checkpoint(checkpoint_path, model, stage1_hash=stage1_hash)
    with torch.cuda.amp.autocast(enabled=False):
        point_features, global_features = stage1_features(encoder, data)
    with torch.no_grad():
        condition = model.encode_condition(
            data.xyz, data.rgb, data.rgb_available, point_features, global_features
        )
        solver = create_scheduler(
            int(cfg.model.diffusion.train_timesteps),
            float(cfg.model.diffusion.beta_start),
            float(cfg.model.diffusion.beta_end),
        )
        prediction = sample(
            model, condition, data.xyz,
            steps=steps or int(cfg.model.diffusion.sample_steps),
            guidance=float(cfg.model.diffusion.guidance),
            seed=int(cfg.seed if seed is None else seed), scheduler=solver,
        )
    output.mkdir(parents=True, exist_ok=True)
    write_point_ply(output / "skeleton.ply", prediction.node_xyz, edges=prediction.edges)
    result = {
        "schema_version": "diffts-stage2-prediction-v1",
        "plant_id": data.plant_id,
        "dataset": data.dataset,
        "view": "top_down",
        "variant": model.variant,
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "predicted_count": prediction.predicted_count,
        "node_xyz": prediction.node_xyz.cpu().tolist(),
        "parent_flow": prediction.parent_flow.cpu().tolist(),
        "parent_index": prediction.parent_index.cpu().tolist(),
        "edges": prediction.edges.cpu().tolist(),
    }
    (output / "skeleton.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args(argv)
    result = run(args.checkpoint, args.input, args.output, seed=args.seed, steps=args.steps)
    print(json.dumps({key: result[key] for key in ("plant_id", "variant", "predicted_count")}, indent=2))


if __name__ == "__main__":
    main()
