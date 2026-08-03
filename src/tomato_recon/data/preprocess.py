"""Deterministic TomatoWUR-to-cache conversion."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from tomato_recon.data.schemas import (
    ORGAN_TYPE_NAMES,
    TOPOLOGY_ROLE_NAMES,
    VISIBILITY_NAMES,
    GraphEdge,
    GraphNode,
    OrganParameters,
    OrganType,
    ParametricPlant,
    PlantGraph,
    PlantSample,
    SemanticClass,
    TopologyRole,
    Visibility,
)
from tomato_recon.data.tomatowur import TomatoWURReader, save_processed_sample


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalise_coordinates(
    xyz: np.ndarray, node_xyz: np.ndarray, root_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = node_xyz[root_index].astype(np.float64)
    normalised_to_original = np.eye(4, dtype=np.float64)
    normalised_to_original[:3, 3] = root
    return (
        (xyz - root).astype(np.float32),
        (node_xyz - root).astype(np.float32),
        normalised_to_original,
    )


def apply_inverse_normalisation(xyz: np.ndarray, normalised_to_original: np.ndarray) -> np.ndarray:
    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=xyz.dtype)], axis=1)
    return (xyz_h @ normalised_to_original.T)[:, :3]


def voxel_downsample(
    xyz: np.ndarray,
    voxel_size_m: float,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    keys = np.floor(xyz / voxel_size_m).astype(np.int64)
    # Lexicographic sorting makes the cache independent of raw row order within a voxel.
    order = np.lexsort((np.arange(len(xyz)), keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    chosen = order[first]
    chosen.sort()
    return xyz[chosen], [array[chosen] for array in arrays], chosen


def _tree_children(parent: np.ndarray) -> tuple[int, list[list[int]]]:
    roots = np.flatnonzero(parent < 0)
    if len(roots) != 1:
        raise ValueError(f"skeleton must have exactly one root, found {len(roots)}")
    root = int(roots[0])
    children = [[] for _ in range(len(parent))]
    for child, value in enumerate(parent):
        if value >= 0:
            if value >= len(parent):
                raise ValueError("skeleton parent index is out of range")
            children[int(value)].append(child)
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            raise ValueError("skeleton contains a cycle")
        seen.add(node)
        stack.extend(children[node])
    if len(seen) != len(parent):
        raise ValueError("skeleton is disconnected")
    return root, children


def resample_skeleton_tree(
    node_xyz: np.ndarray,
    parent_index: np.ndarray,
    edge_type: np.ndarray,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spacing_m <= 0:
        raise ValueError("skeleton_spacing_m must be positive")
    root, children = _tree_children(parent_index)
    new_xyz: list[np.ndarray] = [node_xyz[root]]
    new_parent: list[int] = [-1]
    new_edge_type: list[str] = [""]
    mapping = {root: 0}
    queue = deque([root])
    while queue:
        parent = queue.popleft()
        for child in sorted(children[parent]):
            start, end = node_xyz[parent], node_xyz[child]
            length = float(np.linalg.norm(end - start))
            steps = max(1, int(math.ceil(length / spacing_m)))
            current_parent = mapping[parent]
            for step in range(1, steps + 1):
                alpha = step / steps
                new_xyz.append((1 - alpha) * start + alpha * end)
                new_parent.append(current_parent)
                new_edge_type.append(str(edge_type[child]))
                current_parent = len(new_xyz) - 1
            mapping[child] = current_parent
            queue.append(child)
    return (
        np.asarray(new_xyz, dtype=np.float32),
        np.asarray(new_parent, dtype=np.int64),
        np.asarray(new_edge_type, dtype=object),
    )


def _maximal_chains(parent: np.ndarray, children: list[list[int]], anchors: set[int]) -> list[list[int]]:
    chains: list[list[int]] = []
    for anchor in sorted(anchors):
        for first in sorted(children[anchor]):
            chain = [anchor, first]
            current = first
            while current not in anchors and len(children[current]) == 1:
                current = children[current][0]
                chain.append(current)
            chains.append(chain)
    return chains


def topology_preserving_reduce(
    node_xyz: np.ndarray,
    parent_index: np.ndarray,
    edge_type: np.ndarray,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a rooted tree while retaining root, junctions, tips and connectivity."""
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least two")
    root, children = _tree_children(parent_index)
    if len(node_xyz) <= max_nodes:
        return node_xyz, parent_index, edge_type, np.arange(len(node_xyz), dtype=np.int64)
    anchors = {root}
    anchors.update(i for i, value in enumerate(children) if len(value) != 1)
    if len(anchors) > max_nodes:
        raise ValueError(
            f"max_nodes={max_nodes} cannot preserve {len(anchors)} root/junction/tip nodes; increase K"
        )
    chains = _maximal_chains(parent_index, children, anchors)
    capacities = [max(0, len(chain) - 2) for chain in chains]
    lengths = [
        float(np.linalg.norm(np.diff(node_xyz[chain], axis=0), axis=1).sum()) for chain in chains
    ]
    budget = max_nodes - len(anchors)
    allocations = [0] * len(chains)
    available = set(i for i, capacity in enumerate(capacities) if capacity)
    total_length = sum(lengths[i] for i in available) or float(len(available) or 1)
    weights = {
        i: (lengths[i] / total_length if lengths[i] > 0 else 1.0 / max(len(available), 1))
        for i in available
    }
    while budget > 0 and available:
        # Weighted fair allocation converges to slots proportional to chain arc length.
        index = min(
            available,
            key=lambda i: ((allocations[i] + 1) / max(weights[i], 1e-12), i),
        )
        allocations[index] += 1
        budget -= 1
        available = {i for i in available if allocations[i] < capacities[i]}
    selected = set(anchors)
    for chain, count in zip(chains, allocations, strict=True):
        if count <= 0:
            continue
        internal = np.asarray(chain[1:-1], dtype=np.int64)
        segment = np.linalg.norm(np.diff(node_xyz[chain], axis=0), axis=1)
        cumulative = np.cumsum(segment)[:-1]
        targets = np.linspace(0, segment.sum(), count + 2)[1:-1]
        candidates = []
        for target in targets:
            order = np.argsort(np.abs(cumulative - target), kind="stable")
            candidates.append(next(int(internal[i]) for i in order if int(internal[i]) not in candidates))
        selected.update(candidates)

    # Parent-before-child BFS order produces stable slots and makes reconstruction simple.
    ordered: list[int] = []
    queue = deque([root])
    while queue:
        value = queue.popleft()
        if value in selected:
            ordered.append(value)
        queue.extend(sorted(children[value]))
    old_to_new = {old: new for new, old in enumerate(ordered)}
    reduced_parent = np.full(len(ordered), -1, dtype=np.int64)
    reduced_edge_type = np.full(len(ordered), "", dtype=object)
    for new, old in enumerate(ordered):
        ancestor = int(parent_index[old])
        while ancestor >= 0 and ancestor not in selected:
            ancestor = int(parent_index[ancestor])
        if ancestor >= 0:
            reduced_parent[new] = old_to_new[ancestor]
            reduced_edge_type[new] = edge_type[old]
    _tree_children(reduced_parent)
    return node_xyz[ordered], reduced_parent, reduced_edge_type, np.asarray(ordered)


def fixed_k_skeleton(
    node_xyz: np.ndarray,
    parent_index: np.ndarray,
    edge_type: np.ndarray,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    was_reduced = len(node_xyz) > max_nodes
    reduced_xyz, reduced_parent, reduced_type, kept = topology_preserving_reduce(
        node_xyz, parent_index, edge_type, max_nodes
    )
    count = len(reduced_xyz)
    padded_xyz = np.zeros((max_nodes, 3), dtype=np.float32)
    padded_parent = np.full(max_nodes, -1, dtype=np.int64)
    padded_edge_type = np.full(max_nodes, "", dtype=object)
    valid = np.zeros(max_nodes, dtype=bool)
    padded_xyz[:count] = reduced_xyz
    padded_parent[:count] = reduced_parent
    padded_edge_type[:count] = reduced_type
    valid[:count] = True
    return padded_xyz, padded_parent, padded_edge_type, valid, was_reduced


def _nearest_semantics(node_xyz: np.ndarray, xyz: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    if not len(xyz):
        return np.full(len(node_xyz), int(OrganType.UNKNOWN), dtype=np.int64)
    result = np.empty(len(node_xyz), dtype=np.int64)
    mapping = {
        int(SemanticClass.LEAF): int(OrganType.LEAF_STRUCTURE),
        int(SemanticClass.MAIN_STEM): int(OrganType.MAIN_STEM),
        int(SemanticClass.SIDE_STEM): int(OrganType.SIDE_STEM),
    }
    chunk_size = 256
    for start in range(0, len(node_xyz), chunk_size):
        distances = np.sum((node_xyz[start : start + chunk_size, None] - xyz[None]) ** 2, axis=-1)
        nearest = np.argmin(distances, axis=1)
        result[start : start + chunk_size] = [
            mapping.get(int(semantic[index]), int(OrganType.UNKNOWN)) for index in nearest
        ]
    return result


def derive_node_targets(
    node_xyz: np.ndarray,
    parent_index: np.ndarray,
    node_valid: np.ndarray,
    xyz: np.ndarray,
    semantic: np.ndarray,
    visibility_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = int(node_valid.sum())
    organ = np.full(len(node_valid), int(OrganType.UNKNOWN), dtype=np.int64)
    organ[:count] = _nearest_semantics(node_xyz[:count], xyz, semantic)
    _, children = _tree_children(parent_index[:count])
    role = np.full(len(node_valid), int(TopologyRole.CONTINUATION), dtype=np.int64)
    for index in range(count):
        if parent_index[index] < 0:
            role[index] = int(TopologyRole.ROOT)
        elif not children[index]:
            role[index] = int(TopologyRole.TIP)
        elif len(children[index]) > 1:
            role[index] = int(TopologyRole.JUNCTION)
    visibility = np.full(len(node_valid), int(Visibility.INFERRED_UNKNOWN), dtype=np.int64)
    if len(xyz):
        for start in range(0, count, 256):
            dist2 = np.sum((node_xyz[start : start + 256, None] - xyz[None]) ** 2, axis=-1)
            distance = np.sqrt(dist2.min(axis=1))
            visibility[start : start + len(distance)] = np.where(
                distance <= visibility_distance_m,
                int(Visibility.OBSERVED),
                np.where(
                    distance <= 2 * visibility_distance_m,
                    int(Visibility.PARTIAL),
                    int(Visibility.INFERRED_UNKNOWN),
                ),
            )
    flow = np.zeros_like(node_xyz, dtype=np.float32)
    for child in range(count):
        parent = int(parent_index[child])
        if parent >= 0:
            vector = node_xyz[parent] - node_xyz[child]
            norm = np.linalg.norm(vector)
            flow[child] = vector / max(norm, 1e-8)
    return organ, role, visibility, flow


def graph_from_targets(
    plant_id: str,
    node_xyz: np.ndarray,
    parent_index: np.ndarray,
    node_valid: np.ndarray,
    organ_type: np.ndarray,
    topology_role: np.ndarray,
    visibility: np.ndarray,
    *,
    source: dict[str, Any],
) -> PlantGraph:
    count = int(node_valid.sum())
    root = int(np.flatnonzero(parent_index[:count] < 0)[0])
    nodes = [
        GraphNode(
            id=index,
            xyz=node_xyz[index].tolist(),
            organ_type=ORGAN_TYPE_NAMES[int(organ_type[index])],
            topology_role=TOPOLOGY_ROLE_NAMES[int(topology_role[index])],
            existence_confidence=1.0,
            visibility=VISIBILITY_NAMES[int(visibility[index])],
            source_slot=index,
        )
        for index in range(count)
    ]
    edges = []
    for child in range(count):
        parent = int(parent_index[child])
        if parent >= 0:
            edge_kind = (
                "continuation"
                if organ_type[parent] == organ_type[child] and topology_role[parent] != TopologyRole.JUNCTION
                else "attachment"
            )
            edges.append(GraphEdge(parent=parent, child=child, edge_type=edge_kind, confidence=1.0))
    graph = PlantGraph(
        plant_id=plant_id,
        root_node_id=root,
        nodes=nodes,
        edges=edges,
        source=source,
    )
    graph.validate()
    return graph


def fit_parametric_targets(
    graph: PlantGraph,
    xyz: np.ndarray,
    semantic: np.ndarray,
) -> ParametricPlant:
    children: dict[int, list[int]] = {node.id: [] for node in graph.nodes}
    parent: dict[int, int] = {}
    node_by_id = {node.id: node for node in graph.nodes}
    for edge in graph.edges:
        children[edge.parent].append(edge.child)
        parent[edge.child] = edge.parent
    visited: set[int] = set()
    chains: list[list[int]] = []
    for node in graph.nodes:
        starts = (
            node.id == graph.root_node_id
            or node.id not in parent
            or node_by_id[parent[node.id]].organ_type != node.organ_type
            or len(children[parent[node.id]]) != 1
        )
        if not starts or node.id in visited:
            continue
        chain = [node.id]
        visited.add(node.id)
        current = node.id
        while len(children[current]) == 1:
            nxt = children[current][0]
            if node_by_id[nxt].organ_type != node.organ_type or nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        chains.append(chain)
    chains.extend([[node.id] for node in graph.nodes if node.id not in visited])
    node_to_organ: dict[int, int] = {}
    for organ_id, chain in enumerate(chains):
        for node_id in chain:
            node_to_organ[node_id] = organ_id
    organs: list[OrganParameters] = []
    for organ_id, chain in enumerate(chains):
        organ_type = node_by_id[chain[0]].organ_type
        control = torch.tensor([node_by_id[node_id].xyz for node_id in chain], dtype=torch.float32)
        if len(control) == 1:
            control = torch.cat([control, control + torch.tensor([[0.0, 0.0, 0.005]])])
        transform = torch.eye(4)
        transform[:3, 3] = control[0]
        parent_node = parent.get(chain[0])
        parent_organ = node_to_organ.get(parent_node) if parent_node is not None else None
        distances = np.sqrt(((xyz[:, None] - control.numpy()[None]) ** 2).sum(axis=-1)).min(axis=1)
        if organ_type == "main_stem":
            matching = semantic == int(SemanticClass.MAIN_STEM)
        elif organ_type == "side_stem":
            matching = semantic == int(SemanticClass.SIDE_STEM)
        else:
            matching = semantic == int(SemanticClass.LEAF)
        radius = float(np.clip(np.median(distances[matching]) if matching.any() else 0.003, 0.001, 0.02))
        length = float(torch.linalg.vector_norm(control[1:] - control[:-1], dim=-1).sum())
        confidence = min(node_by_id[node_id].existence_confidence for node_id in chain)
        organs.append(
            OrganParameters(
                organ_id=organ_id,
                organ_type=organ_type,
                parent_organ_id=parent_organ,
                attachment_transform=transform,
                spline_control_points=control,
                radius_start_m=radius if "stem" in organ_type else None,
                radius_end_m=0.75 * radius if "stem" in organ_type else None,
                leaf_length_m=length if organ_type == "leaf_structure" else None,
                leaf_width_coeffs=[0.0, max(length * 0.18, 0.003), 0.0]
                if organ_type == "leaf_structure"
                else None,
                bend_coeffs=[0.0, 0.0] if organ_type == "leaf_structure" else None,
                confidence=confidence,
                source_node_ids=chain,
                visibility=node_by_id[chain[0]].visibility,
            )
        )
    return ParametricPlant(plant_id=graph.plant_id, organs=organs)


def preprocess_record(raw: dict[str, Any], cfg: DictConfig, source: dict[str, Any]) -> tuple[PlantSample, dict[str, Any], np.ndarray]:
    xyz = raw["xyz"]
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError(f"{raw['plant_id']}: XYZ must be non-empty and finite")
    if not np.isfinite(raw["node_xyz"]).all():
        raise ValueError(f"{raw['plant_id']}: skeleton coordinates must be finite")
    xyz, skeleton, transform = normalise_coordinates(xyz, raw["node_xyz"], raw["root_index"])
    support = xyz[raw["semantic"] == int(SemanticClass.SUPPORT_POLE)]
    plant_mask = raw["semantic"] != int(SemanticClass.SUPPORT_POLE)
    if not bool(cfg.remove_support_pole):
        plant_mask[:] = True
    xyz = xyz[plant_mask]
    rgb = raw["rgb"][plant_mask]
    normals = raw["normals"][plant_mask]
    semantic = raw["semantic"][plant_mask]
    instance = raw["instance"][plant_mask]
    xyz, (rgb, normals, semantic, instance), original_indices = voxel_downsample(
        xyz, float(cfg.voxel_size_m), rgb, normals, semantic, instance
    )
    if len(xyz) > int(cfg.num_points):
        selection = np.linspace(0, len(xyz) - 1, int(cfg.num_points), dtype=np.int64)
        xyz, rgb, normals, semantic, instance = (
            value[selection] for value in (xyz, rgb, normals, semantic, instance)
        )
        original_indices = original_indices[selection]
    resampled_xyz, resampled_parent, resampled_edge_type = resample_skeleton_tree(
        skeleton, raw["parent_index"], raw["edge_type"], float(cfg.skeleton_spacing_m)
    )
    pre_reduction_count = len(resampled_xyz)
    node_xyz, parent, _, valid, reduced = fixed_k_skeleton(
        resampled_xyz, resampled_parent, resampled_edge_type, int(cfg.max_nodes)
    )
    organ, role, visibility, flow = derive_node_targets(
        node_xyz,
        parent,
        valid,
        xyz,
        semantic,
        float(cfg.visibility_distance_m),
    )
    metadata = {
        "schema_version": "1.0",
        "dataset": "TomatoWUR-v3",
        "cultivar": raw.get("genotype"),
        "normalised_to_original": transform.tolist(),
        "preprocessing_hash": source["preprocessing_hash"],
        "source_hashes": source["source_hashes"],
        "point_to_original_index": original_indices.tolist(),
        "support_pole_point_count": int(len(support)),
        "traits": {name: values.tolist() for name, values in raw["traits"].items()},
    }
    graph = graph_from_targets(
        raw["plant_id"], node_xyz, parent, valid, organ, role, visibility, source=source
    )
    params = fit_parametric_targets(graph, xyz, semantic)
    sample = PlantSample(
        plant_id=raw["plant_id"],
        xyz=torch.from_numpy(xyz).float(),
        rgb=torch.from_numpy(rgb).float(),
        normals=torch.from_numpy(normals).float(),
        semantic=torch.from_numpy(semantic).long(),
        instance=torch.from_numpy(instance).long(),
        point_valid=torch.ones(len(xyz), dtype=torch.bool),
        node_xyz=torch.from_numpy(node_xyz).float(),
        parent_flow=torch.from_numpy(flow).float(),
        node_valid=torch.from_numpy(valid).bool(),
        parent_index=torch.from_numpy(parent).long(),
        organ_type=torch.from_numpy(organ).long(),
        topology_role=torch.from_numpy(role).long(),
        visibility=torch.from_numpy(visibility).long(),
        graph_target=graph,
        param_target=params,
        metadata=metadata,
    )
    sample.validate()
    stats = {
        "point_count": len(xyz),
        "raw_skeleton_nodes": len(raw["node_xyz"]),
        "resampled_skeleton_nodes": pre_reduction_count,
        "cached_skeleton_nodes": int(valid.sum()),
        "fixed_k_reduced": reduced,
        "support_pole_points": len(support),
    }
    return sample, stats, support


def preprocess_dataset(cfg: DictConfig) -> dict[str, Any]:
    reader = TomatoWURReader(
        cfg.raw_root,
        annotation_version=str(cfg.annotation_version),
        split=str(cfg.split),
        split_file=cfg.get("split_file"),
    )
    output_root = Path(cfg.processed_root)
    samples_dir = output_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    cfg_plain = OmegaConf.to_container(cfg, resolve=True)
    preprocessing_hash = canonical_hash(cfg_plain)
    manifest_samples = []
    warnings: list[str] = []
    reduced_count = 0
    for record in reader:
        source_hashes = {
            "point_cloud": sha256_file(record.point_cloud_path),
            "labels": sha256_file(record.labels_path),
            "skeleton": sha256_file(record.skeleton_path),
        }
        source = {
            "dataset": "TomatoWUR-v3",
            "preprocessing_hash": preprocessing_hash,
            "checkpoint_hashes": {},
            "source_hashes": source_hashes,
        }
        raw = reader.load_record(record)
        sample, stats, support = preprocess_record(raw, cfg, source)
        cache_name = f"{record.plant_id}.npz"
        cache_path = samples_dir / cache_name
        save_processed_sample(sample, cache_path)
        (samples_dir / f"{record.plant_id}.graph.json").write_text(
            json.dumps(sample.graph_target.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (samples_dir / f"{record.plant_id}.params.json").write_text(
            json.dumps(sample.param_target.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if len(support):
            np.savez_compressed(samples_dir / f"{record.plant_id}.context.npz", support_pole_xyz=support)
        reduced_count += int(stats["fixed_k_reduced"])
        manifest_samples.append(
            {
                "plant_id": record.plant_id,
                "cache_file": cache_name,
                "split": str(cfg.split),
                "cache_sha256": sha256_file(cache_path),
                "source_hashes": source_hashes,
                **stats,
            }
        )
    count = len(manifest_samples)
    if count and reduced_count / count > 0.1:
        warnings.append(
            f"{reduced_count}/{count} plants required topology-preserving reduction; review max_nodes"
        )
    manifest = {
        "schema_version": "1.0",
        "dataset": "TomatoWUR-v3",
        "split": str(cfg.split),
        "split_file": str(reader.split_file),
        "preprocessing_config": cfg_plain,
        "preprocessing_hash": preprocessing_hash,
        "label_map": {
            "0": "background",
            "1": "leaf",
            "2": "main_stem",
            "3": "support_pole",
            "4": "side_stem",
        },
        "sample_count": count,
        "fixed_k_reduction_count": reduced_count,
        "fixed_k_reduction_rate": reduced_count / max(count, 1),
        "warnings": warnings,
        "samples": manifest_samples,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
