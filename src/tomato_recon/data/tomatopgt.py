"""Deterministic conversion of complete TomatoPGT scans into canonical samples."""

from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import DictConfig

from tomato_recon.data.conversion import (
    convert_complete_records,
    deterministic_stratified_indices,
    finalise_sample_targets,
    graph_from_tree_targets,
    pad_tree_targets,
)
from tomato_recon.data.preprocess import ProgressCallback, voxel_downsample
from tomato_recon.data.processed import validate_instance_id
from tomato_recon.data.schemas import IGNORE_INDEX, OrganType, PlantSample, SemanticClass, TopologyRole


PGT_SEMANTIC_MAP = {
    0: int(SemanticClass.MAIN_STEM),       # Root-Node
    1: int(SemanticClass.LEAF),            # Compound Leaf-Node
    2: int(SemanticClass.MAIN_STEM),       # Junction-Nodes
    3: int(SemanticClass.LEAF),            # Cotyledon
    6: int(SemanticClass.LEAF),            # Primordium
    7: int(SemanticClass.MAIN_STEM),       # mainStem-Seg
    8: int(SemanticClass.SIDE_STEM),       # Sucker-Seg
    9: int(SemanticClass.SIDE_STEM),       # Stalk-Seg
}
PGT_CLASS_NAMES = {
    "-1": "unannotated",
    "0": "root_node",
    "1": "compound_leaf",
    "2": "junction_node",
    "3": "cotyledon",
    "6": "primordium",
    "7": "main_stem_segment",
    "8": "sucker_segment",
    "9": "stalk_segment",
    "255": "unknown",
}


@dataclass(frozen=True)
class TomatoPGTRecord:
    instance_id: str
    plant_id: str
    split: str
    point_cloud_path: Path
    annotation_path: Path
    graph_path: Path

    @property
    def source_paths(self) -> Mapping[str, Path]:
        return {
            "point_cloud": self.point_cloud_path,
            "annotations": self.annotation_path,
            "graph": self.graph_path,
        }


def _raw_stem(path: Path) -> str:
    return re.sub(r"_R_(?=\d)", "_", path.stem)


def _annotation_stem(path: Path) -> str:
    return path.name.removesuffix("_annotated_ext.txt")


def _graph_stem(path: Path) -> str:
    return path.name.removesuffix("_annotated_ext_graph.json")


class TomatoPGTReader:
    """Discover only source scans having PLY, extended annotations, and a graph."""

    def __init__(self, cfg: DictConfig) -> None:
        self.raw_root = Path(str(cfg.raw_root)).expanduser().resolve()
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"TomatoPGT root does not exist: {self.raw_root}")
        split_map = {str(key): str(value) for key, value in cfg.split_by_source_plant.items()}
        aliases = {str(key): str(value) for key, value in cfg.get("graph_aliases", {}).items()}
        annotations = {
            _annotation_stem(path): path
            for path in self.raw_root.rglob("*_annotated_ext.txt")
        }
        graphs = {_graph_stem(path): path for path in self.raw_root.rglob("*_graph.json")}
        records: list[TomatoPGTRecord] = []
        ignored: list[dict[str, Any]] = []
        used_graphs: set[Path] = set()
        for point_cloud in sorted(self.raw_root.rglob("*.ply")):
            stem = _raw_stem(point_cloud)
            plant_id = point_cloud.parent.parent.name
            annotation = annotations.get(stem)
            graph_stem = aliases.get(stem, stem)
            graph = graphs.get(graph_stem)
            missing = []
            if annotation is None:
                missing.append("annotations")
            if graph is None:
                missing.append("graph")
            if plant_id not in split_map:
                missing.append("split_assignment")
            if missing:
                ignored.append(
                    {
                        "source_instance_id": stem,
                        "source_plant_id": plant_id,
                        "reason": "missing_" + "_and_".join(missing),
                    }
                )
                continue
            assert annotation is not None and graph is not None
            if graph in used_graphs:
                raise ValueError(f"TomatoPGT graph is paired more than once: {graph}")
            used_graphs.add(graph)
            records.append(
                TomatoPGTRecord(
                    instance_id=validate_instance_id(stem),
                    plant_id=plant_id,
                    split=split_map[plant_id],
                    point_cloud_path=point_cloud,
                    annotation_path=annotation,
                    graph_path=graph,
                )
            )
        self.records = sorted(records, key=lambda item: (item.plant_id, item.instance_id))
        self.ignored = ignored

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


_PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_binary_ply_vertices(path: Path) -> dict[str, np.ndarray]:
    """Read named scalar vertex fields from a binary little-endian PLY."""
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii").strip()
        if first != "ply":
            raise ValueError(f"not a PLY file: {path}")
        vertex_count: int | None = None
        element: str | None = None
        properties: list[tuple[str, str]] = []
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"unterminated PLY header: {path}")
            line = raw_line.decode("ascii").strip()
            tokens = line.split()
            if tokens[:1] == ["format"] and tokens[1:2] != ["binary_little_endian"]:
                raise ValueError(f"only binary little-endian PLY is supported: {path}")
            if tokens[:1] == ["element"]:
                element = tokens[1]
                if element == "vertex":
                    vertex_count = int(tokens[2])
            elif tokens[:1] == ["property"] and element == "vertex":
                if tokens[1] == "list":
                    raise ValueError(f"list-valued vertex properties are unsupported: {path}")
                properties.append((tokens[2], tokens[1]))
            elif line == "end_header":
                break
        if vertex_count is None or not properties:
            raise ValueError(f"PLY has no scalar vertex element: {path}")
        try:
            dtype = np.dtype([(name, "<" + _PLY_TYPES[kind]) for name, kind in properties])
        except KeyError as exc:
            raise ValueError(f"unsupported PLY scalar type {exc.args[0]!r}: {path}") from exc
        vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError(f"truncated PLY vertex data: {path}")
    return {name: np.asarray(vertices[name]) for name, _ in properties}


def _fields(fields: Mapping[str, np.ndarray], names: tuple[str, str, str], path: Path) -> np.ndarray:
    missing = [name for name in names if name not in fields]
    if missing:
        raise ValueError(f"PLY is missing fields {missing}: {path}")
    return np.stack([fields[name] for name in names], axis=1).astype(np.float32)


def _sample_polyline(path: np.ndarray, distances: np.ndarray) -> np.ndarray:
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    keep = np.concatenate([[True], np.diff(cumulative) > 1e-10])
    cumulative = cumulative[keep]
    path = path[keep]
    if len(path) == 1:
        return np.repeat(path, len(distances), axis=0)
    result = np.empty((len(distances), 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(distances, cumulative, path[:, axis])
    return result


def _edge_organ(edge_type: str) -> int:
    if edge_type == "STEM":
        return int(OrganType.MAIN_STEM)
    if edge_type in {"STALK", "SUCKER"}:
        return int(OrganType.SIDE_STEM)
    if edge_type in {"CL", "PRIMORDIUM"}:
        return int(OrganType.LEAF_STRUCTURE)
    raise ValueError(f"unsupported TomatoPGT graph edge type: {edge_type!r}")


def resample_tomatopgt_graph(
    graph: Mapping[str, Any], max_nodes: int, target_spacing_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, list[int]]:
    """Convert dense source edge paths to one rooted tree fitting the Stage 1 K."""
    nodes_by_id = {int(node["id"]): node for node in graph["nodes"]}
    roots = [int(node["id"]) for node in graph["nodes"] if node.get("type") == "ROOT"]
    if len(roots) != 1:
        raise ValueError(f"TomatoPGT graph must contain one ROOT node, found {len(roots)}")
    root = roots[0]
    if len(nodes_by_id) > max_nodes:
        raise ValueError(f"source graph has {len(nodes_by_id)} structural nodes but K={max_nodes}")
    adjacency: dict[int, list[tuple[int, Mapping[str, Any]]]] = {
        node_id: [] for node_id in nodes_by_id
    }
    for edge in graph["edges"]:
        source_id, target_id = int(edge["source"]), int(edge["target"])
        if source_id not in adjacency or target_id not in adjacency:
            raise ValueError("TomatoPGT edge references a missing node")
        adjacency[source_id].append((target_id, edge))
        adjacency[target_id].append((source_id, edge))

    source_parent = {root: -1}
    oriented: list[tuple[int, int, Mapping[str, Any]]] = []
    order: list[int] = []
    queue = deque([root])
    while queue:
        parent = queue.popleft()
        order.append(parent)
        for child, edge in sorted(adjacency[parent], key=lambda item: item[0]):
            if child == source_parent[parent]:
                continue
            if child in source_parent:
                raise ValueError("TomatoPGT graph contains a cycle")
            source_parent[child] = parent
            oriented.append((parent, child, edge))
            queue.append(child)
    if len(order) != len(nodes_by_id) or len(oriented) != len(nodes_by_id) - 1:
        raise ValueError("TomatoPGT graph is not a connected tree")

    prepared: list[tuple[int, int, Mapping[str, Any], np.ndarray, float]] = []
    for parent, child, edge in oriented:
        path = np.asarray(edge["path"], dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
            raise ValueError("TomatoPGT graph edge path must have shape [M,3], M>=2")
        parent_xyz = np.asarray(nodes_by_id[parent]["pos"], dtype=np.float64)
        child_xyz = np.asarray(nodes_by_id[child]["pos"], dtype=np.float64)
        forward = np.linalg.norm(path[0] - parent_xyz) + np.linalg.norm(path[-1] - child_xyz)
        reverse = np.linalg.norm(path[-1] - parent_xyz) + np.linalg.norm(path[0] - child_xyz)
        if reverse < forward:
            path = path[::-1].copy()
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        prepared.append((parent, child, edge, path, length))

    def count_at(spacing: float) -> int:
        return len(order) + sum(max(0, math.ceil(length / spacing) - 1) for *_, length in prepared)

    spacing = float(target_spacing_m)
    if spacing <= 0:
        raise ValueError("skeleton_spacing_m must be positive")
    if count_at(spacing) > max_nodes:
        low, high = spacing, max(length for *_, length in prepared)
        for _ in range(64):
            middle = (low + high) / 2
            if count_at(middle) <= max_nodes:
                high = middle
            else:
                low = middle
        spacing = high

    source_to_index = {source_id: index for index, source_id in enumerate(order)}
    xyz = [np.asarray(nodes_by_id[source_id]["pos"], dtype=np.float32) for source_id in order]
    parent_index = np.full(len(order), -1, dtype=np.int64).tolist()
    organ = np.full(len(order), int(OrganType.UNKNOWN), dtype=np.int64).tolist()
    organ[source_to_index[root]] = int(OrganType.MAIN_STEM)
    forced_role = np.full(len(order), -1, dtype=np.int64).tolist()
    forced_role[source_to_index[root]] = int(TopologyRole.ROOT)
    for source_id in order:
        if nodes_by_id[source_id].get("type") == "JUNCTION":
            forced_role[source_to_index[source_id]] = int(TopologyRole.JUNCTION)

    for source_parent_id, source_child_id, edge, path, length in prepared:
        edge_organ = _edge_organ(str(edge["type"]))
        intervals = max(1, math.ceil(length / spacing))
        distances = np.linspace(0.0, length, intervals + 1)[1:-1]
        interior = _sample_polyline(path, distances) if len(distances) else np.empty((0, 3))
        previous = source_to_index[source_parent_id]
        for point in interior:
            xyz.append(point.astype(np.float32))
            parent_index.append(previous)
            organ.append(edge_organ)
            forced_role.append(-1)
            previous = len(xyz) - 1
        child_index = source_to_index[source_child_id]
        parent_index[child_index] = previous
        organ[child_index] = edge_organ

    return (
        np.asarray(xyz, dtype=np.float32),
        np.asarray(parent_index, dtype=np.int64),
        np.asarray(organ, dtype=np.int64),
        np.asarray(forced_role, dtype=np.int64),
        spacing,
        order,
    )


def _normalised_to_original(transform: Mapping[str, Any]) -> np.ndarray:
    translate = np.asarray(transform["translate"], dtype=np.float64)
    rotate = np.asarray(transform["rotate"], dtype=np.float64)
    scale = float(transform["scale_div"])
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = scale * rotate.T
    result[:3, 3] = -translate
    return result


def convert_tomatopgt_record(
    record: TomatoPGTRecord,
    global_id: str,
    source: dict[str, Any],
    cfg: DictConfig,
) -> tuple[PlantSample, dict[str, Any]]:
    fields = read_binary_ply_vertices(record.point_cloud_path)
    xyz_raw = _fields(fields, ("x", "y", "z"), record.point_cloud_path)
    rgb = _fields(fields, ("red", "green", "blue"), record.point_cloud_path) / 255.0
    normals_raw = _fields(fields, ("nx", "ny", "nz"), record.point_cloud_path)
    annotations = np.loadtxt(record.annotation_path, dtype=np.float32)
    if annotations.ndim == 1:
        annotations = annotations[None]
    if annotations.shape[1] < 8:
        raise ValueError(
            "TomatoPGT extended annotation requires at least 8 columns: "
            f"{record.annotation_path}"
        )
    if len(annotations) != len(xyz_raw):
        raise ValueError(
            f"point/annotation row mismatch for {record.instance_id}: "
            f"{len(xyz_raw)} versus {len(annotations)}"
        )
    if not np.allclose(xyz_raw, annotations[:, :3], atol=2e-5, rtol=0):
        raise ValueError(
            "TomatoPGT PLY and annotation rows are not coordinate-aligned: "
            f"{record.instance_id}"
        )

    graph_data = json.loads(record.graph_path.read_text(encoding="utf-8"))
    transform = graph_data.get("meta", {}).get("transform")
    if not isinstance(transform, dict):
        raise ValueError(f"TomatoPGT graph has no coordinate transform: {record.graph_path}")
    translate = np.asarray(transform["translate"], dtype=np.float64)
    rotate = np.asarray(transform["rotate"], dtype=np.float64)
    scale = float(transform["scale_div"])
    if rotate.shape != (3, 3) or scale <= 0:
        raise ValueError(f"invalid TomatoPGT graph transform: {record.graph_path}")
    xyz = ((xyz_raw.astype(np.float64) + translate) @ rotate.T / scale).astype(np.float32)
    normals = (normals_raw.astype(np.float64) @ rotate.T).astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    source_class = annotations[:, 6].astype(np.int64)
    semantic = np.full(len(source_class), IGNORE_INDEX, dtype=np.int64)
    for source_id, target_id in PGT_SEMANTIC_MAP.items():
        semantic[source_class == source_id] = target_id
    unknown = set(np.unique(source_class).tolist()) - set(PGT_SEMANTIC_MAP) - {-1, 255}
    if unknown:
        raise ValueError(f"unsupported TomatoPGT class IDs: {sorted(unknown)}")
    instance = annotations[:, 7].astype(np.int64)
    instance[semantic == IGNORE_INDEX] = -1

    xyz, (rgb, normals, semantic, instance), original_indices = voxel_downsample(
        xyz,
        float(cfg.voxel_size_m),
        rgb.astype(np.float32),
        normals,
        semantic,
        instance,
    )
    selection = deterministic_stratified_indices(semantic, instance, int(cfg.num_points))
    xyz, rgb, normals, semantic, instance = (
        value[selection] for value in (xyz, rgb, normals, semantic, instance)
    )
    original_indices = original_indices[selection]

    raw_nodes, raw_parent, raw_organ, forced_roles, actual_spacing, source_node_ids = (
        resample_tomatopgt_graph(
            graph_data,
            int(cfg.max_nodes),
            float(cfg.skeleton_spacing_m),
        )
    )
    (
        node_xyz,
        parent_index,
        organ_type,
        topology_role,
        visibility,
        parent_flow,
        node_valid,
    ) = pad_tree_targets(
        raw_nodes,
        raw_parent,
        raw_organ,
        int(cfg.max_nodes),
        xyz,
        float(cfg.visibility_distance_m),
        forced_roles=forced_roles,
    )
    graph_source = {
        **source,
        "skeleton_source": "source_derived_cloudgraph",
        "source_graph_file": str(record.graph_path),
    }
    graph = graph_from_tree_targets(
        global_id,
        node_xyz,
        parent_index,
        node_valid,
        organ_type,
        topology_role,
        visibility,
        source=graph_source,
    )
    metadata = {
        "schema_version": "1.0",
        "instance_id": global_id,
        "plant_id": global_id,
        "source_instance_id": record.instance_id,
        "source_plant_id": record.plant_id,
        "dataset": source["dataset"],
        "dataset_name": source["dataset_name"],
        "dataset_version": source["dataset_version"],
        "cultivar": record.plant_id,
        "normalised_to_original": _normalised_to_original(transform).tolist(),
        "preprocessing_hash": source["preprocessing_hash"],
        "source_hashes": source["source_hashes"],
        "point_to_original_index": original_indices.tolist(),
        "rgb_available": True,
        "normals_source": "source_ply",
        "semantic_source": "source_manual_annotation",
        "skeleton_source": "source_derived_cloudgraph",
        "skeleton_modified": True,
        "source_graph_node_ids": source_node_ids,
        "source_graph_path_spacing_m": actual_spacing,
        "source_graph_filename": record.graph_path.name,
    }
    sample = PlantSample(
        plant_id=global_id,
        xyz=torch.from_numpy(xyz).float(),
        rgb=torch.from_numpy(rgb).float(),
        normals=torch.from_numpy(normals).float(),
        semantic=torch.from_numpy(semantic).long(),
        instance=torch.from_numpy(instance).long(),
        point_valid=torch.ones(len(xyz), dtype=torch.bool),
        node_xyz=torch.from_numpy(node_xyz).float(),
        parent_flow=torch.from_numpy(parent_flow).float(),
        node_valid=torch.from_numpy(node_valid).bool(),
        parent_index=torch.from_numpy(parent_index).long(),
        organ_type=torch.from_numpy(organ_type).long(),
        topology_role=torch.from_numpy(topology_role).long(),
        visibility=torch.from_numpy(visibility).long(),
        graph_target=graph,
        metadata=metadata,
    )
    finalise_sample_targets(sample)
    stats = {
        "point_count": len(xyz),
        "raw_skeleton_nodes": len(graph_data["nodes"]),
        "cached_skeleton_nodes": int(node_valid.sum()),
        "skeleton_source": "source_derived_cloudgraph",
        "skeleton_modified": True,
        "skeleton_spacing_m": actual_spacing,
    }
    return sample, stats


def preprocess_tomatopgt(
    cfg: DictConfig, *, progress: ProgressCallback | None = None
) -> dict[str, Any]:
    reader = TomatoPGTReader(cfg)
    expected = int(cfg.get("expected_complete_instances", 0))
    if expected and len(reader) != expected:
        raise ValueError(
            f"expected {expected} complete TomatoPGT instances, discovered {len(reader)}; "
            "check graph aliases and the source archive"
        )
    fields = {
        "label_map": {
            "0": "background",
            "1": "leaf",
            "2": "main_stem",
            "3": "support_pole",
            "4": "side_stem",
        },
        "source_label_map": PGT_CLASS_NAMES,
        "skeleton_source": "source_derived_cloudgraph",
        "skeleton_mode": "source_graph_path_resampled",
        "complete_source_instances_only": True,
    }
    return convert_complete_records(
        cfg,
        reader.records,
        convert_tomatopgt_record,
        dataset_manifest_fields=fields,
        ignored=reader.ignored,
        progress=progress,
    )
