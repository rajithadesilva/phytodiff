"""Tomato-only conversion of annotated Pheno4D scans into canonical samples."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from tomato_recon.data.conversion import (
    convert_complete_records,
    deterministic_stratified_indices,
    finalise_sample_targets,
    graph_from_tree_targets,
    pad_tree_targets,
)
from tomato_recon.data.preprocess import ProgressCallback, voxel_downsample
from tomato_recon.data.processed import validate_instance_id
from tomato_recon.data.schemas import OrganType, PlantSample, SemanticClass


_TOMATO_DIRECTORY = re.compile(r"^Tomato\d{2}$")
_ANNOTATED_SCAN = re.compile(r"^T\d{2}_\d{4}_a\.txt$")


@dataclass(frozen=True)
class Pheno4DRecord:
    instance_id: str
    plant_id: str
    split: str
    point_cloud_path: Path

    @property
    def source_paths(self) -> Mapping[str, Path]:
        return {"annotated_point_cloud": self.point_cloud_path}


class Pheno4DReader:
    """Discover annotated tomato scans and report every unlabelled scan as ignored."""

    def __init__(self, cfg: DictConfig) -> None:
        self.raw_root = Path(str(cfg.raw_root)).expanduser().resolve()
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"Pheno4D root does not exist: {self.raw_root}")
        split_map = {str(key): str(value) for key, value in cfg.split_by_source_plant.items()}
        records: list[Pheno4DRecord] = []
        ignored: list[dict[str, Any]] = []
        for plant_dir in sorted(path for path in self.raw_root.iterdir() if path.is_dir()):
            if not _TOMATO_DIRECTORY.fullmatch(plant_dir.name):
                # Non-tomato data is outside this adapter's scope, not a dataset instance.
                continue
            for path in sorted(plant_dir.glob("*.txt")):
                if _ANNOTATED_SCAN.fullmatch(path.name) and plant_dir.name in split_map:
                    records.append(
                        Pheno4DRecord(
                            instance_id=validate_instance_id(path.stem),
                            plant_id=plant_dir.name,
                            split=split_map[plant_dir.name],
                            point_cloud_path=path,
                        )
                    )
                else:
                    reason = (
                        "missing_annotations"
                        if not path.stem.endswith("_a")
                        else "missing_split_assignment"
                    )
                    ignored.append(
                        {
                            "source_instance_id": path.stem,
                            "source_plant_id": plant_dir.name,
                            "reason": reason,
                        }
                    )
        self.records = sorted(records, key=lambda item: (item.plant_id, item.instance_id))
        self.ignored = ignored

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


def pheno4d_official_orientation(xyz_mm: np.ndarray) -> np.ndarray:
    """Apply the orientation/translation used by the official Pheno4D loader."""
    result = np.empty_like(xyz_mm, dtype=np.float64)
    result[:, 0] = xyz_mm[:, 0] + 50.0
    result[:, 1] = xyz_mm[:, 2]
    result[:, 2] = -xyz_mm[:, 1] - 740.0
    return (result * 0.001).astype(np.float32)


def _resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points.astype(np.float32)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], segment > 1e-8])
    points = points[keep]
    if len(points) < 2:
        return points.astype(np.float32)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    count = max(2, math.ceil(float(cumulative[-1]) / spacing_m) + 1)
    distances = np.linspace(0.0, float(cumulative[-1]), count)
    result = np.stack(
        [np.interp(distances, cumulative, points[:, axis]) for axis in range(3)], axis=1
    )
    return result.astype(np.float32)


def _stem_centerline(stem_xyz: np.ndarray, slice_m: float) -> np.ndarray:
    if len(stem_xyz) < 10:
        raise ValueError("Pheno4D scan has too few stem points to reconstruct a centreline")
    minimum, maximum = float(stem_xyz[:, 2].min()), float(stem_xyz[:, 2].max())
    if maximum - minimum < slice_m:
        # Very young tomato plants can have a complete but near-horizontal labelled
        # stem. In that case Z slicing collapses the centreline, so slice along the
        # stem's dominant axis and orient the result from its lower endpoint.
        centred = stem_xyz.astype(np.float64) - stem_xyz.mean(axis=0, keepdims=True)
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        coordinate = centred @ axes[0]
        minimum, maximum = float(coordinate.min()), float(coordinate.max())
        if maximum - minimum < 1e-5:
            raise ValueError("Pheno4D stem is too compact to reconstruct a centreline")
    else:
        coordinate = stem_xyz[:, 2].astype(np.float64)
    bins = max(2, math.ceil((maximum - minimum) / slice_m))
    edges = np.linspace(minimum, maximum + 1e-9, bins + 1)
    centres: list[np.ndarray] = []
    for index in range(bins):
        mask = (coordinate >= edges[index]) & (coordinate < edges[index + 1])
        if mask.any():
            centres.append(np.median(stem_xyz[mask], axis=0))
    result = np.asarray(centres, dtype=np.float64)
    if len(result) < 2:
        raise ValueError("Pheno4D stem slicing produced fewer than two centreline points")
    # Axis bins are already ordered; flip only when their first endpoint is above
    # the last one so node zero remains the biologically plausible basal root.
    if result[0, 2] > result[-1, 2]:
        result = result[::-1].copy()
    if len(result) > 2:
        smooth = result.copy()
        smooth[1:-1, :2] = (
            result[:-2, :2] + 2.0 * result[1:-1, :2] + result[2:, :2]
        ) / 4.0
        result = smooth
    return result.astype(np.float32)


def _leaf_path_fallback(points: np.ndarray, base: np.ndarray) -> np.ndarray:
    centred = points - points.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    direction = axes[0]
    projection = (points - base) @ direction
    if abs(float(projection.min())) > abs(float(projection.max())):
        direction = -direction
        projection = -projection
    endpoint = points[int(np.argmax(projection))]
    return np.stack([base, endpoint]).astype(np.float32)


def _leaf_geodesic_path(
    points: np.ndarray,
    stem_path: np.ndarray,
    *,
    voxel_m: float,
    neighbors: int,
    radius_m: float,
    max_points: int,
) -> np.ndarray:
    reduced, _, _ = voxel_downsample(points, voxel_m)
    if len(reduced) > max_points:
        reduced = reduced[np.linspace(0, len(reduced) - 1, max_points, dtype=np.int64)]
    if len(reduced) < 4:
        base = points[cKDTree(stem_path).query(points)[0].argmin()]
        return _leaf_path_fallback(points, base)
    stem_tree = cKDTree(stem_path)
    distance_to_stem, _ = stem_tree.query(reduced, k=1)
    base_index = int(np.argmin(distance_to_stem))
    base = reduced[base_index]
    tree = cKDTree(reduced)
    k = min(max(2, neighbors + 1), len(reduced))
    distances, indices = tree.query(reduced, k=k)
    rows = np.repeat(np.arange(len(reduced)), k - 1)
    cols = indices[:, 1:].reshape(-1)
    weights = distances[:, 1:].reshape(-1)
    keep = np.isfinite(weights) & (weights <= radius_m)
    adjacency = coo_matrix(
        (weights[keep], (rows[keep], cols[keep])), shape=(len(reduced), len(reduced))
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    graph_distance, predecessors = dijkstra(
        adjacency, directed=False, indices=base_index, return_predecessors=True
    )
    reachable = np.flatnonzero(np.isfinite(graph_distance))
    if len(reachable) < max(4, len(reduced) // 10):
        return _leaf_path_fallback(points, base)
    endpoint = int(reachable[np.argmax(graph_distance[reachable])])
    path_indices = [endpoint]
    while path_indices[-1] != base_index:
        previous = int(predecessors[path_indices[-1]])
        if previous < 0 or previous in path_indices:
            return _leaf_path_fallback(points, base)
        path_indices.append(previous)
    path = reduced[np.asarray(path_indices[::-1], dtype=np.int64)]
    if len(path) > 2:
        smooth = path.copy()
        smooth[1:-1] = (path[:-2] + 2.0 * path[1:-1] + path[2:]) / 4.0
        path = smooth
    return path.astype(np.float32)


def reconstruct_pheno4d_tree(
    xyz: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    stem = xyz[semantic == int(SemanticClass.MAIN_STEM)]
    stem_path = _stem_centerline(stem, float(cfg.stem_slice_m))
    leaf_paths: list[np.ndarray] = []
    for leaf_id in sorted(
        int(value)
        for value in np.unique(instance[semantic == int(SemanticClass.LEAF)])
        if value >= 2
    ):
        leaf = xyz[(semantic == int(SemanticClass.LEAF)) & (instance == leaf_id)]
        if len(leaf) < int(cfg.min_leaf_points):
            continue
        leaf_paths.append(
            _leaf_geodesic_path(
                leaf,
                stem_path,
                voxel_m=float(cfg.leaf_skeleton_voxel_m),
                neighbors=int(cfg.leaf_graph_neighbors),
                radius_m=float(cfg.leaf_graph_radius_m),
                max_points=int(cfg.leaf_graph_max_points),
            )
        )

    def build(spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stem_nodes = _resample_polyline(stem_path, spacing)
        nodes = [point for point in stem_nodes]
        parents = [-1] + list(range(len(stem_nodes) - 1))
        organs = [int(OrganType.MAIN_STEM)] * len(stem_nodes)
        for leaf_path in leaf_paths:
            leaf_nodes = _resample_polyline(leaf_path, spacing)
            if len(leaf_nodes) < 2:
                continue
            attachment = int(
                np.linalg.norm(stem_nodes - leaf_nodes[0][None], axis=1).argmin()
            )
            previous = attachment
            for point in leaf_nodes:
                nodes.append(point)
                parents.append(previous)
                organs.append(int(OrganType.LEAF_STRUCTURE))
                previous = len(nodes) - 1
        return (
            np.asarray(nodes, dtype=np.float32),
            np.asarray(parents, dtype=np.int64),
            np.asarray(organs, dtype=np.int64),
        )

    spacing = float(cfg.skeleton_spacing_m)
    if spacing <= 0:
        raise ValueError("skeleton_spacing_m must be positive")
    nodes, parents, organs = build(spacing)
    while len(nodes) > int(cfg.max_nodes):
        spacing *= 1.1
        nodes, parents, organs = build(spacing)
        if spacing > 0.25:
            raise ValueError(
                f"Pheno4D reconstructed tree cannot fit K={int(cfg.max_nodes)} "
                f"while preserving {len(leaf_paths)} leaves"
            )
    return nodes, parents, organs, spacing, len(leaf_paths)


def estimate_normals(
    xyz: np.ndarray,
    semantic: np.ndarray,
    *,
    neighbors: int,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Estimate deterministic PCA normals and orient them by plant structure."""
    if len(xyz) < 3:
        return np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (len(xyz), 1))
    tree = cKDTree(xyz)
    k = min(max(3, neighbors), len(xyz))
    _, neighbor_indices = tree.query(xyz, k=k)
    normals = np.empty_like(xyz, dtype=np.float32)
    for start in range(0, len(xyz), chunk_size):
        stop = min(start + chunk_size, len(xyz))
        local = xyz[neighbor_indices[start:stop]].astype(np.float64)
        centred = local - local.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centred, centred) / max(k - 1, 1)
        _, vectors = np.linalg.eigh(covariance)
        normals[start:stop] = vectors[:, :, 0].astype(np.float32)

    stem = xyz[semantic == int(SemanticClass.MAIN_STEM)]
    centre_xy = np.median(stem[:, :2], axis=0) if len(stem) else np.median(xyz[:, :2], axis=0)
    radial = np.zeros_like(normals)
    radial[:, :2] = xyz[:, :2] - centre_xy
    radial_dot = np.sum(normals * radial, axis=1)
    background = semantic == int(SemanticClass.BACKGROUND)
    leaf = semantic == int(SemanticClass.LEAF)
    vertical_choice = background | (leaf & (np.abs(normals[:, 2]) >= 0.2))
    flip = np.where(vertical_choice, normals[:, 2] < 0, radial_dot < 0)
    normals[flip] *= -1
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = norms[:, 0] < 1e-8
    normals /= np.maximum(norms, 1e-8)
    normals[degenerate] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


def _normalised_to_source_mm(root_oriented_m: np.ndarray) -> np.ndarray:
    rotation = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
    )
    translation_mm = np.asarray([50.0, 0.0, -740.0], dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = 1000.0 * rotation.T
    result[:3, 3] = rotation.T @ (1000.0 * root_oriented_m - translation_mm)
    return result


def convert_pheno4d_record(
    record: Pheno4DRecord,
    global_id: str,
    source: dict[str, Any],
    cfg: DictConfig,
) -> tuple[PlantSample, dict[str, Any]]:
    values = np.loadtxt(record.point_cloud_path, dtype=np.float32)
    if values.ndim == 1:
        values = values[None]
    if values.shape[1] != 4:
        raise ValueError(
            f"annotated Pheno4D tomato files must have four columns, got {values.shape[1]}: "
            f"{record.point_cloud_path}"
        )
    xyz = pheno4d_official_orientation(values[:, :3])
    labels = values[:, 3].astype(np.int64)
    if labels.min(initial=0) < 0:
        raise ValueError(f"Pheno4D labels must be non-negative: {record.point_cloud_path}")
    semantic = np.where(
        labels == 0,
        int(SemanticClass.BACKGROUND),
        np.where(labels == 1, int(SemanticClass.MAIN_STEM), int(SemanticClass.LEAF)),
    ).astype(np.int64)
    instance = np.where(labels == 0, -1, labels).astype(np.int64)
    xyz, (semantic, instance), original_indices = voxel_downsample(
        xyz, float(cfg.voxel_size_m), semantic, instance
    )
    selection = deterministic_stratified_indices(semantic, instance, int(cfg.num_points))
    xyz, semantic, instance = (value[selection] for value in (xyz, semantic, instance))
    original_indices = original_indices[selection]

    raw_nodes, raw_parent, raw_organ, actual_spacing, leaf_count = reconstruct_pheno4d_tree(
        xyz, semantic, instance, cfg
    )
    root_oriented = raw_nodes[int(np.flatnonzero(raw_parent < 0)[0])].copy()
    xyz = (xyz - root_oriented).astype(np.float32)
    raw_nodes = (raw_nodes - root_oriented).astype(np.float32)
    normals = estimate_normals(xyz, semantic, neighbors=int(cfg.normal_neighbors))
    rgb = np.zeros((len(xyz), 3), dtype=np.float32)
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
    )
    graph_source = {
        **source,
        "skeleton_source": "reconstructed_from_manual_organs",
        "reconstruction_method": "stem_slices_and_leaf_geodesics_v1",
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
        confidence=float(cfg.reconstructed_graph_confidence),
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
        "normalised_to_original": _normalised_to_source_mm(root_oriented).tolist(),
        "source_coordinate_unit": "millimetres",
        "preprocessing_hash": source["preprocessing_hash"],
        "source_hashes": source["source_hashes"],
        "point_to_original_index": original_indices.tolist(),
        "rgb_available": False,
        "rgb_imputation": "zeros",
        "normals_source": "local_pca",
        "semantic_source": "source_manual_annotation",
        "skeleton_source": "reconstructed_from_manual_organs",
        "skeleton_reconstruction_method": "stem_slices_and_leaf_geodesics_v1",
        "skeleton_modified": True,
        "reconstructed_leaf_count": leaf_count,
        "skeleton_spacing_m": actual_spacing,
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
        "raw_skeleton_nodes": len(raw_nodes),
        "cached_skeleton_nodes": int(node_valid.sum()),
        "skeleton_source": "reconstructed_from_manual_organs",
        "skeleton_modified": True,
        "skeleton_spacing_m": actual_spacing,
        "reconstructed_leaf_count": leaf_count,
    }
    return sample, stats


def preprocess_pheno4d(
    cfg: DictConfig, *, progress: ProgressCallback | None = None
) -> dict[str, Any]:
    reader = Pheno4DReader(cfg)
    expected = int(cfg.get("expected_complete_instances", 0))
    if expected and len(reader) != expected:
        raise ValueError(
            f"expected {expected} annotated Pheno4D tomato instances, discovered {len(reader)}"
        )
    fields = {
        "label_map": {
            "0": "background",
            "1": "leaf",
            "2": "main_stem",
            "3": "support_pole",
            "4": "side_stem",
        },
        "source_label_map": {
            "0": "soil_background",
            "1": "stem",
            ">=2": "individual_leaf",
        },
        "skeleton_source": "reconstructed_from_manual_organs",
        "skeleton_mode": "stem_slices_and_leaf_geodesics_v1",
        "rgb_source": "zero_imputed_unavailable",
        "normals_source": "local_pca",
        "complete_source_instances_only": True,
        "species_filter": "tomato",
    }
    return convert_complete_records(
        cfg,
        reader.records,
        convert_pheno4d_record,
        dataset_manifest_fields=fields,
        ignored=reader.ignored,
        progress=progress,
    )
