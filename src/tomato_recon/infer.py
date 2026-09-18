from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tomato_recon.config import load_config
from tomato_recon.data.collate import collate_plant_samples
from tomato_recon.data.schemas import IGNORE_INDEX, OrganType, PlantSample, TopologyRole, Visibility
from tomato_recon.data.processed import load_processed_sample, normalise_point_cloud_types
from tomato_recon.data.side import load_side_sample
from tomato_recon.data.top_down import load_top_down_sample
from tomato_recon.export.ply import write_mesh_ply, write_point_ply
from tomato_recon.export.usd_exporter import USDPlantExporter
from tomato_recon.models.pipeline import PipelineResult, TomatoReconstructionPipeline
from tomato_recon.train.common import load_checkpoint, seed_everything, select_device
from tomato_recon.train.common import checkpoint_sha256


SEMANTIC_COLOURS = torch.tensor(
    [[40, 40, 40], [40, 170, 55], [35, 110, 35], [110, 110, 110], [80, 145, 45]],
    dtype=torch.float32,
)


def _load_ascii_ply(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "ply" or "format ascii" not in lines[1]:
        raise ValueError("only ASCII PLY input is supported by the lightweight inference reader")
    count = next(int(line.split()[-1]) for line in lines if line.startswith("element vertex"))
    end = lines.index("end_header")
    return np.asarray([[float(v) for v in line.split()[:3]] for line in lines[end + 1 : end + 1 + count]], dtype=np.float32)


def load_unlabelled_scan(path: Path, max_nodes: int, num_points: int) -> PlantSample:
    if path.suffix.lower() == ".csv":
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
        names = {name.lower(): name for name in data.dtype.names or ()}
        xyz = np.stack([data[names[axis]] for axis in ("x", "y", "z")], axis=-1).astype(np.float32)
        rgb = np.stack(
            [data[names[channel]] if channel in names else np.zeros(len(data)) for channel in ("red", "green", "blue")],
            axis=-1,
        ).astype(np.float32)
        if rgb.max(initial=0) > 1:
            rgb /= 255
        normals = np.stack(
            [data[names[channel]] if channel in names else np.zeros(len(data)) for channel in ("nx", "ny", "nz")],
            axis=-1,
        ).astype(np.float32)
    elif path.suffix.lower() == ".ply":
        xyz = _load_ascii_ply(path)
        rgb = np.zeros_like(xyz)
        normals = np.zeros_like(xyz)
    else:
        raise ValueError("inference input must be a processed .npz, TomatoWUR-style .csv, or ASCII .ply")
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("input scan must contain finite XYZ points")
    root = np.array([np.median(xyz[:, 0]), np.median(xyz[:, 1]), xyz[:, 2].min()], dtype=np.float32)
    xyz = xyz - root
    if len(xyz) > num_points:
        keep = np.linspace(0, len(xyz) - 1, num_points, dtype=np.int64)
        xyz, rgb, normals = xyz[keep], rgb[keep], normals[keep]
    node_xyz = torch.zeros((max_nodes, 3))
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = root
    return PlantSample(
        plant_id=path.stem,
        xyz=torch.from_numpy(xyz),
        rgb=torch.from_numpy(rgb),
        normals=torch.from_numpy(normals),
        semantic=torch.full((len(xyz),), IGNORE_INDEX, dtype=torch.long),
        instance=torch.full((len(xyz),), -1, dtype=torch.long),
        point_valid=torch.ones(len(xyz), dtype=torch.bool),
        node_xyz=node_xyz,
        parent_flow=torch.zeros_like(node_xyz),
        node_valid=torch.zeros(max_nodes, dtype=torch.bool),
        parent_index=torch.full((max_nodes,), -1, dtype=torch.long),
        organ_type=torch.full((max_nodes,), int(OrganType.UNKNOWN), dtype=torch.long),
        topology_role=torch.full((max_nodes,), int(TopologyRole.CONTINUATION), dtype=torch.long),
        visibility=torch.full((max_nodes,), int(Visibility.INFERRED_UNKNOWN), dtype=torch.long),
        metadata={
            "dataset": "user_scan",
            "preprocessing_hash": None,
            "normalised_to_original": transform.tolist(),
        },
    )


def _preview(path: Path, sample: PlantSample, result: PipelineResult) -> None:
    size = 640
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    values = torch.cat([sample.xyz[:, [0, 2]], torch.tensor([node.xyz for node in result.graph.nodes])[:, [0, 2]]])
    minimum, maximum = values.amin(0), values.amax(0)
    scale = (size - 40) / float((maximum - minimum).max().clamp_min(1e-5))

    def point(value: torch.Tensor) -> tuple[float, float]:
        shifted = (value - minimum) * scale + 20
        return float(shifted[0]), float(size - shifted[1])

    for value in sample.xyz[:: max(1, len(sample.xyz) // 3000), [0, 2]]:
        x, y = point(value)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(160, 190, 160))
    node = {item.id: torch.tensor(item.xyz)[[0, 2]] for item in result.graph.nodes}
    for edge in result.graph.edges:
        draw.line((point(node[edge.parent]), point(node[edge.child])), fill=(20, 70, 20), width=3)
    for item in result.graph.nodes:
        x, y = point(node[item.id])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(200, 40, 20))
    image.save(path)


def write_inference_outputs(
    output_dir: Path, sample: PlantSample, result: PipelineResult, cfg
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic = result.semantic_logits[: len(sample.xyz)].argmax(-1).cpu()
    write_point_ply(output_dir / "input_normalised.ply", sample.xyz, sample.rgb)
    write_point_ply(output_dir / "semantic_prediction.ply", sample.xyz, SEMANTIC_COLOURS[semantic])
    graph_xyz = torch.tensor([node.xyz for node in result.graph.nodes])
    graph_edges = torch.tensor([[edge.parent, edge.child] for edge in result.graph.edges], dtype=torch.long)
    write_point_ply(output_dir / "skeleton_prediction.ply", graph_xyz, edges=graph_edges)
    skeleton_json = {
        "schema_version": "1.0",
        "plant_id": sample.plant_id,
        "nodes": [node.to_dict() for node in result.graph.nodes],
        "sample_id": int(result.skeleton.sample_id[0]) if result.skeleton.sample_id is not None else None,
    }
    (output_dir / "skeleton_prediction.json").write_text(
        json.dumps(skeleton_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "plant_graph.json").write_text(
        json.dumps(result.graph.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "organ_parameters.json").write_text(
        json.dumps(result.parameters.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_mesh_ply(output_dir / "reconstructed_mesh.ply", result.geometry.combined_mesh())
    (output_dir / "traits.json").write_text(
        json.dumps(result.traits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "uncertainty.json").write_text(
        json.dumps(result.uncertainty, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if bool(cfg.export.usd):
        report = USDPlantExporter(
            meters_per_unit=float(cfg.export.meters_per_unit), up_axis=str(cfg.export.up_axis)
        ).export(
            result.graph,
            result.geometry,
            output_dir / "plant.usd",
            write_debug_usda=bool(cfg.export.write_debug_usda),
            include_skeleton=bool(cfg.export.include_skeleton),
            include_collisions=bool(cfg.export.include_collisions),
            include_context_pole=bool(cfg.export.include_context_pole),
        )
        report_dict = report.to_dict()
    else:
        report_dict = {"valid": True, "skipped": True, "reason": "export.usd=false"}
    (output_dir / "export_report.json").write_text(
        json.dumps(report_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _preview(output_dir / "preview.png", sample, result)
    return {path.name: str(path) for path in sorted(output_dir.iterdir())}


def main(argv: list[str] | None = None) -> None:
    cfg, _ = load_config("infer", argv)
    if not cfg.input.path:
        raise ValueError("set input.path to a processed sample .npz, raw .csv, or ASCII .ply")
    input_path = Path(cfg.input.path)
    pcl_types = normalise_point_cloud_types(cfg.data.pcl_types)
    processed_input = input_path.suffix.lower() == ".npz"
    if processed_input:
        full_sample = load_processed_sample(input_path)
        samples = {
            view: (
                replace(
                    full_sample,
                    metadata={**full_sample.metadata, "pcl_type": "full"},
                )
                if view == "full"
                else load_top_down_sample(input_path, full_sample)
                if view == "top_down"
                else load_side_sample(input_path, full_sample)
            )
            for view in pcl_types
        }
    else:
        if pcl_types != ("full",):
            raise ValueError("raw CSV/PLY inference requires data.pcl_types=[full]")
        full_sample = load_unlabelled_scan(
            input_path, int(cfg.data.max_nodes), int(cfg.data.num_points)
        )
        samples = {"full": replace(
            full_sample, metadata={**full_sample.metadata, "pcl_type": "full"}
        )}
    if len(full_sample.node_xyz) != int(cfg.data.max_nodes):
        raise ValueError(
            f"input fixed-K={len(full_sample.node_xyz)} does not match configured "
            f"data.max_nodes={cfg.data.max_nodes}"
        )
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    pipeline = TomatoReconstructionPipeline(cfg).to(device).eval()
    checkpoint_path = Path(cfg.model.pipeline_checkpoint)
    if checkpoint_path.is_file():
        checkpoint = load_checkpoint(
            checkpoint_path,
            pipeline,
            expected_preprocessing_hash=full_sample.metadata.get("preprocessing_hash"),
            expected_max_nodes=int(cfg.data.max_nodes),
        )
        pipeline.checkpoint_hashes = {"pipeline": checkpoint_sha256(checkpoint_path)}
        pipeline.git_commit = str(checkpoint.get("git_commit", ""))
    elif not bool(cfg.trainer.fast_dev_run):
        raise FileNotFoundError(f"pipeline checkpoint not found: {checkpoint_path}")
    output_root = Path(cfg.output.dir)
    outputs: dict[str, dict[str, str]] = {}
    for view in pcl_types:
        sample = samples[view]
        batch = collate_plant_samples([sample]).to(device)
        result = pipeline.reconstruct(batch)[0]
        output_dir = output_root if len(pcl_types) == 1 else output_root / view
        outputs[view] = write_inference_outputs(output_dir, sample, result, cfg)
    if len(pcl_types) > 1:
        output_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "1.0",
            "plant_id": full_sample.plant_id,
            "input": str(input_path),
            "pcl_types": list(pcl_types),
            "views": {
                view: {
                    "output": str(output_root / view),
                    "point_count": int(samples[view].point_valid.sum()),
                    "outputs": outputs[view],
                }
                for view in pcl_types
            },
        }
        (output_root / "inference_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "plant_id": full_sample.plant_id,
                "pcl_types": list(pcl_types),
                "outputs": outputs[pcl_types[0]] if len(pcl_types) == 1 else outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
