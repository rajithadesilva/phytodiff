"""Deterministic TomatoWUR-to-cache conversion."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

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
from tomato_recon.data.processed import (
    load_processed_dataset_manifest,
    next_plant_number,
    plant_instance_id,
    processed_dataset_identity,
    resolve_processed_dataset_root,
    save_processed_sample,
    write_processed_dataset_manifest,
)
from tomato_recon.data.tomatowur import TomatoWURReader


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(
    callback: ProgressCallback | None, phase: str, **details: Any
) -> None:
    if callback is not None:
        callback({"phase": phase, **details})


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


def pad_ground_truth_skeleton(
    node_xyz: np.ndarray,
    parent_index: np.ndarray,
    edge_type: np.ndarray,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad an official skeleton to K without changing any GT node or edge."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    count = len(node_xyz)
    if len(parent_index) != count or len(edge_type) != count:
        raise ValueError("skeleton node, parent, and edge-type arrays must have equal length")
    _tree_children(parent_index)
    if count > max_nodes:
        raise ValueError(
            f"official ground-truth skeleton has {count} nodes but K={max_nodes}; "
            "increase max_nodes because GT resampling or reduction is disabled"
        )
    padded_xyz = np.zeros((max_nodes, 3), dtype=np.float32)
    padded_parent = np.full(max_nodes, -1, dtype=np.int64)
    padded_edge_type = np.full(max_nodes, "", dtype=object)
    valid = np.zeros(max_nodes, dtype=bool)
    padded_xyz[:count] = node_xyz
    padded_parent[:count] = parent_index
    padded_edge_type[:count] = edge_type
    valid[:count] = True
    return padded_xyz, padded_parent, padded_edge_type, valid


def derive_node_targets(
    node_xyz: np.ndarray,
    parent_index: np.ndarray,
    edge_type: np.ndarray,
    node_valid: np.ndarray,
    xyz: np.ndarray,
    visibility_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = int(node_valid.sum())
    organ = np.full(len(node_valid), int(OrganType.UNKNOWN), dtype=np.int64)
    root, children = _tree_children(parent_index[:count])
    organ[root] = int(OrganType.MAIN_STEM)
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in children[parent]:
            relation = str(edge_type[child])
            if relation not in {"<", "+"}:
                raise ValueError(
                    f"unsupported official skeleton edge type {relation!r} at node {child}"
                )
            organ[child] = (
                int(OrganType.SIDE_STEM)
                if relation == "+" or organ[parent] == int(OrganType.SIDE_STEM)
                else int(OrganType.MAIN_STEM)
            )
            stack.append(child)
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
    annotated_edge_type: np.ndarray,
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
                "continuation" if str(annotated_edge_type[child]) == "<" else "attachment"
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
        radius = float(
            np.clip(np.median(distances[matching]) if matching.any() else 0.003, 0.001, 0.02)
        )
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


def preprocess_record(
    raw: dict[str, Any], cfg: DictConfig, source: dict[str, Any]
) -> tuple[PlantSample, dict[str, Any], np.ndarray]:
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
    node_xyz, parent, edge_type, valid = pad_ground_truth_skeleton(
        skeleton,
        np.asarray(raw["parent_index"], dtype=np.int64),
        np.asarray(raw["edge_type"], dtype=object),
        int(cfg.max_nodes),
    )
    organ, role, visibility, flow = derive_node_targets(
        node_xyz,
        parent,
        edge_type,
        valid,
        xyz,
        float(cfg.visibility_distance_m),
    )
    node_ids = np.asarray(raw["node_ids"], dtype=np.int64)
    official_parent_ids = [
        int(node_ids[parent_index]) if int(parent_index) >= 0 else None
        for parent_index in raw["parent_index"]
    ]
    metadata = {
        "schema_version": "1.0",
        "instance_id": raw["instance_id"],
        "plant_id": raw["plant_id"],
        "source_instance_id": raw["source_instance_id"],
        "source_plant_id": raw["source_plant_id"],
        "dataset": source["dataset"],
        "dataset_name": source["dataset_name"],
        "dataset_version": source["dataset_version"],
        "cultivar": raw.get("genotype"),
        "normalised_to_original": transform.tolist(),
        "preprocessing_hash": source["preprocessing_hash"],
        "source_hashes": source["source_hashes"],
        "point_to_original_index": original_indices.tolist(),
        "support_pole_point_count": int(len(support)),
        "skeleton_source": "official_ground_truth",
        "skeleton_annotation_version": source["annotation_version"],
        "skeleton_modified": False,
        "official_gt_node_ids": node_ids.tolist(),
        "official_gt_parent_ids": official_parent_ids,
        "official_gt_edge_types": [str(value) for value in raw["edge_type"]],
        "traits": {name: values.tolist() for name, values in raw["traits"].items()},
    }
    graph = graph_from_targets(
        raw["plant_id"],
        node_xyz,
        parent,
        valid,
        organ,
        role,
        visibility,
        edge_type,
        source=source,
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
        "cached_skeleton_nodes": int(valid.sum()),
        "support_pole_points": len(support),
        "skeleton_source": "official_ground_truth",
        "skeleton_modified": False,
    }
    return sample, stats, support


def preprocess_dataset(
    cfg: DictConfig, *, progress: ProgressCallback | None = None
) -> dict[str, Any]:
    dataset_identity = processed_dataset_identity(cfg)
    output_root = resolve_processed_dataset_root(cfg)
    if str(cfg.get("skeleton_mode", "")) != "official_gt_direct":
        raise ValueError(
            "data.skeleton_mode must be 'official_gt_direct'; heuristic skeleton "
            "repair, resampling, and reduction are no longer supported"
        )
    splits = [str(value) for value in cfg.get("splits", [str(cfg.split)])]
    if not splits or len(splits) != len(set(splits)):
        raise ValueError("data.splits must contain one or more unique split names")
    if str(cfg.split) not in splits:
        raise ValueError("data.split must be included in data.splits")
    _emit_progress(
        progress,
        "loading_splits",
        dataset=dataset_identity["dataset"],
        raw_root=str(cfg.raw_root),
        output_root=str(output_root),
        splits=splits,
    )
    configured_split_files = cfg.get("split_files", {})
    readers: dict[str, TomatoWURReader] = {}
    for split in splits:
        split_file = configured_split_files.get(split) if configured_split_files else None
        if split_file is None and len(splits) == 1:
            split_file = cfg.get("split_file")
        readers[split] = TomatoWURReader(
            cfg.raw_root,
            annotation_version=str(cfg.annotation_version),
            split=split,
            split_file=split_file,
        )
    total_instances = sum(len(reader) for reader in readers.values())
    _emit_progress(
        progress,
        "discovered",
        total=total_instances,
        split_counts={split: len(reader) for split, reader in readers.items()},
    )
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_manifest = load_processed_dataset_manifest(output_root)
    cfg_plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_plain, dict):
        raise TypeError("preprocessing configuration must be a mapping")
    # Split orchestration does not alter an individual sample's preprocessing
    # transforms, so cache compatibility depends only on the transform settings.
    hash_config = dict(cfg_plain)
    hash_config.pop("splits", None)
    hash_config.pop("split_files", None)
    preprocessing_hash = canonical_hash(hash_config)
    dataset_manifest["datasets"][dataset_identity["dataset"]] = {
        **dataset_identity,
        "preprocessing_config": cfg_plain,
        "preprocessing_hash": preprocessing_hash,
        "label_map": {
            "0": "background",
            "1": "leaf",
            "2": "main_stem",
            "3": "support_pole",
            "4": "side_stem",
        },
        "split_files": {split: str(reader.split_file) for split, reader in readers.items()},
        "skeleton_source": "official_ground_truth",
        "skeleton_mode": "official_gt_direct",
        "skeleton_annotation_version": str(cfg.annotation_version),
    }
    source_index: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in dataset_manifest["instances"]:
        key = (str(entry.get("dataset", "")), str(entry.get("source_instance_id", "")))
        if key in source_index:
            raise ValueError(f"duplicate source instance in processed manifest: {key}")
        source_index[key] = entry
    next_number = next_plant_number(output_root, dataset_manifest)
    manifest_instances = []
    warnings: list[str] = []
    seen_instance_ids: set[str] = set()
    seen_point_clouds: set[Path] = set()
    plant_splits = {
        str(entry["source_plant_id"]): str(entry["split"])
        for entry in dataset_manifest["instances"]
        if entry.get("dataset") == dataset_identity["dataset"]
        and entry.get("source_plant_id")
    }
    current_plant_ids: set[str] = set()
    split_counts = {split: 0 for split in splits}
    new_instance_count = 0
    resumed_instance_count = 0
    skipped_instance_count = 0
    current_index = 0
    for split, reader in readers.items():
        for record in reader:
            current_index += 1
            started_at = time.perf_counter()
            progress_details = {
                "current": current_index,
                "total": total_instances,
                "split": split,
                "source_instance_id": record.instance_id,
            }
            _emit_progress(progress, "checking", **progress_details)
            if record.instance_id in seen_instance_ids:
                raise ValueError(
                    f"instance ID {record.instance_id!r} occurs more than once in the "
                    "configured splits"
                )
            point_cloud_path = record.point_cloud_path.resolve()
            if point_cloud_path in seen_point_clouds:
                raise ValueError(
                    f"point cloud {point_cloud_path} is assigned to more than one instance"
                )
            previous_split = plant_splits.get(record.plant_id)
            if previous_split is not None and previous_split != split:
                raise ValueError(
                    f"plant ID {record.plant_id!r} has point-cloud instances in both "
                    f"{previous_split!r} and {split!r}; split plants rather than scans"
                )
            seen_instance_ids.add(record.instance_id)
            seen_point_clouds.add(point_cloud_path)
            plant_splits[record.plant_id] = split
            current_plant_ids.add(record.plant_id)
            source_hashes = {
                "point_cloud": sha256_file(record.point_cloud_path),
                "labels": sha256_file(record.labels_path),
                "skeleton": sha256_file(record.skeleton_path),
            }
            source_key = (dataset_identity["dataset"], record.instance_id)
            manifest_entry = source_index.get(source_key)
            existing_instance = manifest_entry is not None
            if manifest_entry is None:
                global_instance_id = plant_instance_id(next_number)
                next_number += 1
                cache_relative = Path(global_instance_id) / "sample.npz"
                manifest_entry = {
                    **dataset_identity,
                    "instance_id": global_instance_id,
                    "plant_number": int(global_instance_id.removeprefix("plant_")),
                    "source_instance_id": record.instance_id,
                    "plant_id": global_instance_id,
                    "source_plant_id": record.plant_id,
                    "point_cloud_id": record.point_cloud_path.stem,
                    "cache_file": cache_relative.as_posix(),
                    "split": split,
                    "preprocessing_hash": preprocessing_hash,
                    "source_hashes": source_hashes,
                    "status": "processing",
                }
                dataset_manifest["instances"].append(manifest_entry)
                source_index[source_key] = manifest_entry
                new_instance_count += 1
            else:
                global_instance_id = str(manifest_entry["instance_id"])
                cache_relative = Path(str(manifest_entry["cache_file"]))
            instance_dir = output_root / global_instance_id
            cache_path = output_root / cache_relative
            graph_path = cache_path.with_suffix(".graph.json")
            params_path = cache_path.with_suffix(".params.json")
            unchanged = (
                manifest_entry.get("status") == "complete"
                and manifest_entry.get("preprocessing_hash") == preprocessing_hash
                and manifest_entry.get("source_hashes") == source_hashes
                and cache_path.is_file()
                and graph_path.is_file()
                and params_path.is_file()
            )
            if unchanged:
                split_counts[split] += 1
                skipped_instance_count += 1
                manifest_instances.append(dict(manifest_entry))
                _emit_progress(
                    progress,
                    "skipped",
                    **progress_details,
                    instance_id=global_instance_id,
                    elapsed_seconds=time.perf_counter() - started_at,
                )
                continue
            if existing_instance:
                resumed_instance_count += 1
            action = "resuming" if existing_instance else "creating"
            _emit_progress(
                progress,
                "processing",
                **progress_details,
                instance_id=global_instance_id,
                action=action,
            )
            manifest_entry.update(
                {
                    **dataset_identity,
                    "source_instance_id": record.instance_id,
                    "plant_id": global_instance_id,
                    "source_plant_id": record.plant_id,
                    "point_cloud_id": record.point_cloud_path.stem,
                    "cache_file": cache_relative.as_posix(),
                    "split": split,
                    "preprocessing_hash": preprocessing_hash,
                    "source_hashes": source_hashes,
                    "status": "processing",
                }
            )
            write_processed_dataset_manifest(output_root, dataset_manifest)
            source = {
                **dataset_identity,
                "instance_id": global_instance_id,
                "source_instance_id": record.instance_id,
                "source_plant_id": record.plant_id,
                "split": split,
                "annotation_version": str(cfg.annotation_version),
                "preprocessing_hash": preprocessing_hash,
                "checkpoint_hashes": {},
                "source_hashes": source_hashes,
            }
            raw = reader.load_record(record)
            raw["source_instance_id"] = raw["instance_id"]
            raw["source_plant_id"] = raw["plant_id"]
            raw["instance_id"] = global_instance_id
            raw["plant_id"] = global_instance_id
            sample, stats, support = preprocess_record(raw, cfg, source)
            sample.metadata["split"] = split
            _emit_progress(
                progress,
                "writing",
                **progress_details,
                instance_id=global_instance_id,
                action=action,
            )
            instance_dir.mkdir(parents=True, exist_ok=True)
            # Remove reports written by the retired heuristic repair pipeline.
            quality_path = instance_dir / "quality.json"
            quality_path.unlink(missing_ok=True)
            save_processed_sample(sample, cache_path)
            graph_path.write_text(
                json.dumps(sample.graph_target.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            params_path.write_text(
                json.dumps(sample.param_target.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            context_path = instance_dir / "context.npz"
            if len(support):
                np.savez_compressed(
                    context_path,
                    support_pole_xyz=support,
                )
            else:
                context_path.unlink(missing_ok=True)
            split_counts[split] += 1
            manifest_entry.update(
                {
                    "cache_sha256": sha256_file(cache_path),
                    "status": "complete",
                    **stats,
                }
            )
            manifest_instances.append(dict(manifest_entry))
            write_processed_dataset_manifest(output_root, dataset_manifest)
            _emit_progress(
                progress,
                "completed",
                **progress_details,
                instance_id=global_instance_id,
                action=action,
                point_count=stats["point_count"],
                elapsed_seconds=time.perf_counter() - started_at,
            )
    count = len(manifest_instances)
    report = {
        "schema_version": "1.0",
        "layout": "flat-plant-instance-v1",
        **dataset_identity,
        "split": str(cfg.split),
        "splits": splits,
        "split_file": str(readers[str(cfg.split)].split_file),
        "split_files": {split: str(reader.split_file) for split, reader in readers.items()},
        "split_counts": split_counts,
        "source_split_counts": {split: len(reader) for split, reader in readers.items()},
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
        "instance_count": count,
        "plant_count": count,
        "source_plant_count": len(current_plant_ids),
        "point_cloud_count": count,
        "new_instance_count": new_instance_count,
        "resumed_instance_count": resumed_instance_count,
        "skipped_instance_count": skipped_instance_count,
        "next_plant_number": next_plant_number(output_root, dataset_manifest),
        "source_sample_count": sum(len(reader) for reader in readers.values()),
        "skeleton_source": "official_ground_truth",
        "skeleton_mode": "official_gt_direct",
        "skeleton_annotation_version": str(cfg.annotation_version),
        "skeleton_modified_count": 0,
        "warnings": warnings,
        "instances": manifest_instances,
    }
    write_processed_dataset_manifest(output_root, dataset_manifest)
    _emit_progress(
        progress,
        "finished",
        total=total_instances,
        new_instance_count=new_instance_count,
        resumed_instance_count=resumed_instance_count,
        skipped_instance_count=skipped_instance_count,
        next_plant_number=report["next_plant_number"],
    )
    return report
