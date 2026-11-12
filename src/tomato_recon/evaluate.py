from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tomato_recon.data.schemas import PlantGraph
from tomato_recon.data.processed import ProcessedPlantDataset
from tomato_recon.evaluation.graph_metrics import graph_metrics
from tomato_recon.evaluation.geometry_metrics import geometry_metrics
from tomato_recon.evaluation.skeleton_metrics import skeleton_metrics
from tomato_recon.evaluation.trait_metrics import trait_metrics


def _read_ascii_ply(path: Path, limit: int = 4096) -> tuple[torch.Tensor, torch.Tensor | None]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3 or lines[0] != "ply" or "format ascii" not in lines[1]:
        raise ValueError(f"evaluation supports ASCII PLY geometry only: {path}")
    count = next(int(line.split()[-1]) for line in lines if line.startswith("element vertex"))
    header_end = lines.index("end_header")
    properties = []
    for line in lines[:header_end]:
        if line.startswith("element vertex"):
            properties = []
        elif line.startswith("element ") and properties:
            break
        elif line.startswith("property ") and "list" not in line:
            properties.append(line.split()[-1])
    values = torch.tensor(
        [[float(value) for value in line.split()[: len(properties)]] for line in lines[header_end + 1 : header_end + 1 + count]],
        dtype=torch.float32,
    )
    index = torch.arange(len(values))
    if len(values) > limit:
        index = torch.linspace(0, len(values) - 1, limit).long()
    column = {name: position for position, name in enumerate(properties)}
    points = values[index][:, [column[axis] for axis in ("x", "y", "z")]]
    normals = None
    if all(axis in column for axis in ("nx", "ny", "nz")):
        normals = values[index][:, [column[axis] for axis in ("nx", "ny", "nz")]]
    return points, normals


def graph_traits(graph: PlantGraph) -> dict[str, float]:
    nodes = {node.id: torch.tensor(node.xyz) for node in graph.nodes}
    lengths = [float(torch.linalg.vector_norm(nodes[e.child] - nodes[e.parent])) for e in graph.edges]
    heights = torch.tensor([node.xyz[2] for node in graph.nodes])
    return {
        "plant_height_m": float(heights.max() - heights.min()),
        "total_skeleton_length_m": sum(lengths),
        "mean_internode_length_m": sum(lengths) / max(len(lengths), 1),
        "node_count": float(len(graph.nodes)),
        "leaf_structure_count": float(sum(node.organ_type == "leaf_structure" for node in graph.nodes)),
    }


def evaluate_directory(processed_root: Path, predictions: Path) -> dict:
    dataset = ProcessedPlantDataset(processed_root)
    per_sample = {}
    for sample in dataset:
        graph_path = predictions / sample.plant_id / "plant_graph.json"
        if not graph_path.is_file() and len(dataset) == 1:
            graph_path = predictions / "plant_graph.json"
        if not graph_path.is_file():
            per_sample[sample.plant_id] = {"error": f"missing prediction {graph_path}"}
            continue
        predicted = PlantGraph.from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
        if sample.graph_target is None:
            per_sample[sample.plant_id] = {"error": "processed sample has no graph target"}
            continue
        predicted_xyz = torch.tensor([node.xyz for node in predicted.nodes])
        target_xyz = sample.node_xyz[sample.node_valid]
        geometry_path = graph_path.parent / "reconstructed_mesh.ply"
        if geometry_path.is_file():
            model_points, model_normals = _read_ascii_ply(geometry_path)
            scan_points = sample.xyz[sample.point_valid]
            scan_normals = sample.normals[sample.point_valid]
            if len(scan_points) > 4096:
                keep = torch.linspace(0, len(scan_points) - 1, 4096).long()
                scan_points, scan_normals = scan_points[keep], scan_normals[keep]
            geometry_result: dict = geometry_metrics(
                model_points,
                scan_points,
                model_normals=model_normals,
                scan_normals=scan_normals if model_normals is not None else None,
            )
            geometry_result["surface_sample_limit"] = 4096
        else:
            geometry_result = {
                "status": "unavailable",
                "note": f"missing reconstructed geometry {geometry_path}",
            }
        per_sample[sample.plant_id] = {
            "skeleton": skeleton_metrics(predicted_xyz, target_xyz),
            "topology": graph_metrics(predicted, sample.graph_target),
            "traits": trait_metrics(graph_traits(predicted), graph_traits(sample.graph_target)),
            "geometry": geometry_result,
            "fruit": {
                "status": "excluded_from_primary_metrics",
                "predicted_count": sum(n.organ_type == "fruit_optional" for n in predicted.nodes),
            },
        }
    aggregate: dict[str, float] = {}
    flat: dict[str, list[float]] = {}
    secondary_flat: dict[str, list[float]] = {}
    for result in per_sample.values():
        if "error" in result:
            continue
        for section in ("skeleton", "topology"):
            for key, value in result[section].items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    flat.setdefault(f"{section}.{key}", []).append(float(value))
        for trait, errors in result["traits"].items():
            for key, value in errors.items():
                if math.isfinite(float(value)):
                    flat.setdefault(f"traits.{trait}.{key}", []).append(float(value))
        for section in ("geometry", "fruit"):
            for key, value in result[section].items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    secondary_flat.setdefault(f"{section}.{key}", []).append(float(value))
    for key, values in flat.items():
        aggregate[key] = sum(values) / len(values)
    return {
        "schema_version": "1.0",
        "primary_metric_groups": ["skeleton", "topology", "traits"],
        "secondary_metric_groups": ["geometry", "fruit"],
        "aggregate": aggregate,
        "secondary_aggregate": {
            key: sum(values) / len(values) for key, values in secondary_flat.items()
        },
        "per_sample": per_sample,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate cached reconstruction predictions")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/metrics.json"))
    args = parser.parse_args(argv)
    result = evaluate_directory(args.processed_root, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
