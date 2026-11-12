"""Standalone AG-GFM inference; no target arrays or files are read."""

import argparse
import json
from pathlib import Path

from tomato_recon.data.aggfm import load_input, sample_points
from tomato_recon.export.ply import write_point_ply
from tomato_recon.models.flow_matching.runtime import check_preprocessing, load_runtime, predict


def run(checkpoint, input_path, output, *, view="full", steps=None, seed=None, dataset=None):
    model, encoder, cfg, payload, device = load_runtime(checkpoint)
    data = load_input(input_path, view)
    if dataset is not None and dataset != "combined" and data.dataset != dataset:
        raise ValueError("input dataset does not match the selected inference dataset")
    check_preprocessing({"datasets": {data.dataset: [data.metadata.get("preprocessing_hash")]}},
                         payload["dataset_compatibility"])
    data = sample_points(data, int(cfg.model.aggfm.max_points), int(cfg.seed if seed is None else seed)).to(device)
    prediction, seeds = predict(model, encoder, data, cfg, steps=steps, seed=seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_point_ply(output / "skeleton.ply", prediction.node_xyz, edges=prediction.edges)
    result = {"schema_version": "aggfm-stage2-prediction-v1", "stage2_model": "aggfm",
              "plant_id": data.plant_id, "dataset": data.dataset, "view": data.view,
              "checkpoint": str(checkpoint), "input": str(input_path),
              "predicted_count": prediction.predicted_count,
              "node_xyz": prediction.node_xyz.cpu().tolist(),
              "parent_index": prediction.parent_index.cpu().tolist(),
              "parent_flow": prediction.parent_flow.cpu().tolist(), "edges": prediction.edges.cpu().tolist(),
              "source_slots": prediction.source_slots.cpu().tolist(), "graph": prediction.graph.to_dict(),
              "observed_seeds": seeds.state[seeds.observed, :3].cpu().tolist(),
              "latent_seeds": seeds.state[~seeds.observed, :3].cpu().tolist()}
    (output / "skeleton.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "graph.json").write_text(json.dumps(result["graph"], indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=("full", "top_down", "side"), default="full")
    parser.add_argument("--dataset", choices=("combined", "tomatowur", "tomatopgt", "pheno4d"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    run(args.checkpoint, args.input, args.output, view=args.view, steps=args.steps, seed=args.seed,
        dataset=args.dataset)


if __name__ == "__main__":
    main()
