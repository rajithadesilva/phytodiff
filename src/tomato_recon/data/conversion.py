"""Shared orchestration and target helpers for canonical dataset conversions."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from omegaconf import DictConfig, OmegaConf

from tomato_recon.data.preprocess import (
    ProgressCallback,
    _emit_progress,
    canonical_hash,
    fit_parametric_targets,
    sha256_file,
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
from tomato_recon.data.schemas import (
    ORGAN_TYPE_NAMES,
    TOPOLOGY_ROLE_NAMES,
    VISIBILITY_NAMES,
    GraphEdge,
    GraphNode,
    PlantGraph,
    PlantSample,
    TopologyRole,
    Visibility,
)
from tomato_recon.data.side import SideSettings, ensure_side
from tomato_recon.data.top_down import TopDownSettings, ensure_top_down


class CompleteSourceRecord(Protocol):
    instance_id: str
    plant_id: str
    split: str
    point_cloud_path: Path

    @property
    def source_paths(self) -> Mapping[str, Path]: ...


RecordConverter = Callable[
    [CompleteSourceRecord, str, dict[str, Any], DictConfig],
    tuple[PlantSample, dict[str, Any]],
]


def deterministic_stratified_indices(
    semantic: np.ndarray,
    instance: np.ndarray,
    limit: int,
) -> np.ndarray:
    """Select deterministic proportional samples without dropping small organ groups."""
    if limit < 1:
        raise ValueError("num_points must be positive")
    if len(semantic) != len(instance):
        raise ValueError("semantic and instance arrays must have equal length")
    if len(semantic) <= limit:
        return np.arange(len(semantic), dtype=np.int64)

    keys = np.stack([semantic.astype(np.int64), instance.astype(np.int64)], axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    groups = len(counts)
    if groups > limit:
        # This should not occur for the plant datasets, but remains deterministic.
        return np.linspace(0, len(semantic) - 1, limit, dtype=np.int64)

    exact = counts.astype(np.float64) * float(limit) / float(len(semantic))
    allocation = np.maximum(1, np.floor(exact).astype(np.int64))
    while int(allocation.sum()) > limit:
        candidates = np.flatnonzero(allocation > 1)
        remove = candidates[np.argmax(allocation[candidates] - exact[candidates])]
        allocation[remove] -= 1
    while int(allocation.sum()) < limit:
        candidates = np.flatnonzero(allocation < counts)
        add = candidates[np.argmax(exact[candidates] - allocation[candidates])]
        allocation[add] += 1

    selected: list[np.ndarray] = []
    for group, count in enumerate(allocation):
        indices = np.flatnonzero(inverse == group)
        if count >= len(indices):
            selected.append(indices)
        else:
            selected.append(indices[np.linspace(0, len(indices) - 1, count, dtype=np.int64)])
    return np.sort(np.concatenate(selected)).astype(np.int64)


def pad_tree_targets(
    node_xyz: np.ndarray,
    parent_index: np.ndarray,
    organ_type: np.ndarray,
    max_nodes: int,
    xyz: np.ndarray,
    visibility_distance_m: float,
    *,
    forced_roles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate a rooted tree, derive roles/visibility/flow, and pad it to K."""
    count = len(node_xyz)
    if not (len(parent_index) == len(organ_type) == count):
        raise ValueError("tree target arrays must have equal length")
    if count < 1 or count > max_nodes:
        raise ValueError(f"tree contains {count} nodes but K={max_nodes}")
    roots = np.flatnonzero(parent_index < 0)
    if len(roots) != 1:
        raise ValueError(f"tree must have exactly one root, found {len(roots)}")
    children = [[] for _ in range(count)]
    for child, parent in enumerate(parent_index):
        if parent >= count:
            raise ValueError("tree parent index is out of range")
        if parent >= 0:
            children[int(parent)].append(child)
    seen: set[int] = set()
    stack = [int(roots[0])]
    while stack:
        node = stack.pop()
        if node in seen:
            raise ValueError("tree contains a cycle")
        seen.add(node)
        stack.extend(children[node])
    if len(seen) != count:
        raise ValueError("tree is disconnected")

    role = np.full(count, int(TopologyRole.CONTINUATION), dtype=np.int64)
    for index in range(count):
        if parent_index[index] < 0:
            role[index] = int(TopologyRole.ROOT)
        elif not children[index]:
            role[index] = int(TopologyRole.TIP)
        elif len(children[index]) > 1:
            role[index] = int(TopologyRole.JUNCTION)
    if forced_roles is not None:
        forced_roles = np.asarray(forced_roles, dtype=np.int64)
        if forced_roles.shape != (count,):
            raise ValueError("forced topology roles must match the node count")
        mask = forced_roles >= 0
        role[mask] = forced_roles[mask]
        role[int(roots[0])] = int(TopologyRole.ROOT)

    visibility = np.full(count, int(Visibility.INFERRED_UNKNOWN), dtype=np.int64)
    if len(xyz):
        for start in range(0, count, 128):
            distances2 = np.sum((node_xyz[start : start + 128, None] - xyz[None]) ** 2, axis=-1)
            distance = np.sqrt(distances2.min(axis=1))
            visibility[start : start + len(distance)] = np.where(
                distance <= visibility_distance_m,
                int(Visibility.OBSERVED),
                np.where(
                    distance <= 2 * visibility_distance_m,
                    int(Visibility.PARTIAL),
                    int(Visibility.INFERRED_UNKNOWN),
                ),
            )

    flow = np.zeros((count, 3), dtype=np.float32)
    for child, parent in enumerate(parent_index):
        if parent >= 0:
            vector = node_xyz[int(parent)] - node_xyz[child]
            flow[child] = vector / max(float(np.linalg.norm(vector)), 1e-8)

    padded_xyz = np.zeros((max_nodes, 3), dtype=np.float32)
    padded_parent = np.full(max_nodes, -1, dtype=np.int64)
    padded_organ = np.full(max_nodes, 4, dtype=np.int64)
    padded_role = np.full(max_nodes, int(TopologyRole.CONTINUATION), dtype=np.int64)
    padded_visibility = np.full(max_nodes, int(Visibility.INFERRED_UNKNOWN), dtype=np.int64)
    padded_flow = np.zeros((max_nodes, 3), dtype=np.float32)
    valid = np.zeros(max_nodes, dtype=bool)
    padded_xyz[:count] = node_xyz
    padded_parent[:count] = parent_index
    padded_organ[:count] = organ_type
    padded_role[:count] = role
    padded_visibility[:count] = visibility
    padded_flow[:count] = flow
    valid[:count] = True
    return (
        padded_xyz,
        padded_parent,
        padded_organ,
        padded_role,
        padded_visibility,
        padded_flow,
        valid,
    )


def graph_from_tree_targets(
    plant_id: str,
    node_xyz: np.ndarray,
    parent_index: np.ndarray,
    node_valid: np.ndarray,
    organ_type: np.ndarray,
    topology_role: np.ndarray,
    visibility: np.ndarray,
    *,
    source: dict[str, Any],
    confidence: float = 1.0,
) -> PlantGraph:
    count = int(node_valid.sum())
    root = int(np.flatnonzero(parent_index[:count] < 0)[0])
    nodes = [
        GraphNode(
            id=index,
            xyz=node_xyz[index].tolist(),
            organ_type=ORGAN_TYPE_NAMES[int(organ_type[index])],
            topology_role=TOPOLOGY_ROLE_NAMES[int(topology_role[index])],
            existence_confidence=confidence,
            visibility=VISIBILITY_NAMES[int(visibility[index])],
            source_slot=index,
        )
        for index in range(count)
    ]
    edges: list[GraphEdge] = []
    for child in range(count):
        parent = int(parent_index[child])
        if parent < 0:
            continue
        edge_type = (
            "continuation"
            if int(organ_type[parent]) == int(organ_type[child])
            else "attachment"
        )
        edges.append(
            GraphEdge(parent=parent, child=child, edge_type=edge_type, confidence=confidence)
        )
    graph = PlantGraph(
        plant_id=plant_id,
        root_node_id=root,
        nodes=nodes,
        edges=edges,
        source=source,
    )
    graph.validate()
    return graph


def finalise_sample_targets(sample: PlantSample) -> PlantSample:
    """Fit the derived parametric targets after a converter has built its graph."""
    if sample.graph_target is None:
        raise ValueError("converted sample must provide a graph target")
    sample.param_target = fit_parametric_targets(
        sample.graph_target,
        sample.xyz.cpu().numpy(),
        sample.semantic.cpu().numpy(),
    )
    sample.validate()
    return sample


def convert_complete_records(
    cfg: DictConfig,
    records: Sequence[CompleteSourceRecord],
    converter: RecordConverter,
    *,
    dataset_manifest_fields: Mapping[str, Any],
    ignored: Sequence[Mapping[str, Any]],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Write complete source records into the shared flat plant-instance dataset."""
    identity = processed_dataset_identity(cfg)
    top_down_settings = TopDownSettings.from_config(cfg)
    side_settings = SideSettings.from_config(cfg)
    output_root = resolve_processed_dataset_root(cfg)
    cfg_plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_plain, dict):
        raise TypeError("preprocessing configuration must be a mapping")
    hash_config = {
        key: value for key, value in cfg_plain.items() if key not in {"top_down", "side"}
    }
    preprocessing_hash = canonical_hash(hash_config)
    records = sorted(records, key=lambda item: (item.plant_id, item.instance_id))
    source_ids = [record.instance_id for record in records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{identity['dataset']} contains duplicate source instance IDs")
    source_point_clouds = [record.point_cloud_path.resolve() for record in records]
    if len(source_point_clouds) != len(set(source_point_clouds)):
        raise ValueError(f"{identity['dataset']} assigns one point cloud to multiple instances")
    split_counts_discovered: dict[str, int] = {}
    for record in records:
        split_counts_discovered[record.split] = split_counts_discovered.get(record.split, 0) + 1
    _emit_progress(
        progress,
        "loading_splits",
        dataset=identity["dataset"],
        raw_root=str(cfg.raw_root),
        output_root=str(output_root),
        splits=sorted(split_counts_discovered),
    )
    _emit_progress(
        progress,
        "discovered",
        total=len(records),
        split_counts=split_counts_discovered,
        ignored_count=len(ignored),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_processed_dataset_manifest(output_root)
    manifest["datasets"][identity["dataset"]] = {
        **identity,
        "preprocessing_config": cfg_plain,
        "preprocessing_hash": preprocessing_hash,
        "ignored_incomplete_count": len(ignored),
        "ignored_incomplete": [dict(item) for item in ignored],
        **dict(dataset_manifest_fields),
    }
    source_index: dict[tuple[str, str], dict[str, Any]] = {
        (str(entry.get("dataset", "")), str(entry.get("source_instance_id", ""))): entry
        for entry in manifest["instances"]
    }
    if len(source_index) != len(manifest["instances"]):
        raise ValueError("processed manifest contains duplicate source instances")
    plant_splits = {
        (str(entry.get("dataset", "")), str(entry.get("source_plant_id", ""))): str(
            entry.get("split", "")
        )
        for entry in manifest["instances"]
        if entry.get("source_plant_id")
    }
    next_number = next_plant_number(output_root, manifest)
    new_count = resumed_count = skipped_count = 0
    completed_entries: list[dict[str, Any]] = []

    for current, record in enumerate(records, start=1):
        started_at = time.perf_counter()
        details = {
            "current": current,
            "total": len(records),
            "split": record.split,
            "source_instance_id": record.instance_id,
        }
        _emit_progress(progress, "checking", **details)
        split_key = (identity["dataset"], record.plant_id)
        previous_split = plant_splits.get(split_key)
        if previous_split is not None and previous_split != record.split:
            raise ValueError(
                f"source plant {record.plant_id!r} occurs in both {previous_split!r} "
                f"and {record.split!r}; split physical plants rather than scans"
            )
        plant_splits[split_key] = record.split
        source_hashes = {
            name: sha256_file(path) for name, path in sorted(record.source_paths.items())
        }
        source_key = (identity["dataset"], record.instance_id)
        entry = source_index.get(source_key)
        existing = entry is not None
        if entry is None:
            global_id = plant_instance_id(next_number)
            next_number += 1
            entry = {
                **identity,
                "instance_id": global_id,
                "plant_number": int(global_id.removeprefix("plant_")),
                "source_instance_id": record.instance_id,
                "plant_id": global_id,
                "source_plant_id": record.plant_id,
                "point_cloud_id": record.point_cloud_path.stem,
                "cache_file": f"{global_id}/sample.npz",
                "split": record.split,
                "preprocessing_hash": preprocessing_hash,
                "source_hashes": source_hashes,
                "status": "processing",
            }
            manifest["instances"].append(entry)
            source_index[source_key] = entry
            new_count += 1
        else:
            global_id = str(entry["instance_id"])
        cache_path = output_root / str(entry["cache_file"])
        graph_path = cache_path.with_suffix(".graph.json")
        params_path = cache_path.with_suffix(".params.json")
        unchanged = (
            entry.get("status") == "complete"
            and entry.get("preprocessing_hash") == preprocessing_hash
            and entry.get("source_hashes") == source_hashes
            and cache_path.is_file()
            and graph_path.is_file()
            and params_path.is_file()
        )
        if unchanged:
            top_down_action = ensure_top_down(output_root, entry, top_down_settings)
            side_action = ensure_side(output_root, entry, side_settings)
            if "generated" in {top_down_action, side_action}:
                write_processed_dataset_manifest(output_root, manifest)
            if top_down_action == "generated":
                _emit_progress(
                    progress, "top_down", **details,
                    instance_id=global_id, point_count=entry["top_down"]["point_count"],
                )
            if side_action == "generated":
                _emit_progress(
                    progress, "side", **details,
                    instance_id=global_id, point_count=entry["side"]["point_count"],
                )
            skipped_count += 1
            completed_entries.append(dict(entry))
            _emit_progress(
                progress,
                "skipped",
                **details,
                instance_id=global_id,
                elapsed_seconds=time.perf_counter() - started_at,
            )
            continue
        if existing:
            resumed_count += 1
        action = "resuming" if existing else "creating"
        entry.update(
            {
                **identity,
                "plant_id": global_id,
                "source_plant_id": record.plant_id,
                "point_cloud_id": record.point_cloud_path.stem,
                "split": record.split,
                "preprocessing_hash": preprocessing_hash,
                "source_hashes": source_hashes,
                "status": "processing",
            }
        )
        write_processed_dataset_manifest(output_root, manifest)
        _emit_progress(
            progress,
            "processing",
            **details,
            instance_id=global_id,
            action=action,
        )
        source = {
            **identity,
            "instance_id": global_id,
            "source_instance_id": record.instance_id,
            "source_plant_id": record.plant_id,
            "split": record.split,
            "preprocessing_hash": preprocessing_hash,
            "checkpoint_hashes": {},
            "source_hashes": source_hashes,
        }
        sample, stats = converter(record, global_id, source, cfg)
        sample.metadata["split"] = record.split
        _emit_progress(
            progress,
            "writing",
            **details,
            instance_id=global_id,
            action=action,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_processed_sample(sample, cache_path)
        assert sample.graph_target is not None and sample.param_target is not None
        graph_path.write_text(
            json.dumps(sample.graph_target.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        params_path.write_text(
            json.dumps(sample.param_target.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ensure_top_down(output_root, entry, top_down_settings)
        ensure_side(output_root, entry, side_settings)
        entry.update({"cache_sha256": sha256_file(cache_path), "status": "complete", **stats})
        completed_entries.append(dict(entry))
        write_processed_dataset_manifest(output_root, manifest)
        _emit_progress(
            progress,
            "completed",
            **details,
            instance_id=global_id,
            action=action,
            point_count=stats["point_count"],
            elapsed_seconds=time.perf_counter() - started_at,
        )

    split_counts: dict[str, int] = {}
    for entry in completed_entries:
        split = str(entry["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
    report = {
        "schema_version": "1.0",
        "layout": "flat-plant-instance-v1",
        **identity,
        "preprocessing_config": cfg_plain,
        "preprocessing_hash": preprocessing_hash,
        "sample_count": len(completed_entries),
        "instance_count": len(completed_entries),
        "plant_count": len(completed_entries),
        "source_plant_count": len({entry["source_plant_id"] for entry in completed_entries}),
        "point_cloud_count": len(completed_entries),
        "new_instance_count": new_count,
        "resumed_instance_count": resumed_count,
        "skipped_instance_count": skipped_count,
        "ignored_incomplete_count": len(ignored),
        "ignored_incomplete": list(ignored),
        "split_counts": split_counts,
        "next_plant_number": next_plant_number(output_root, manifest),
        "instances": completed_entries,
        **dict(dataset_manifest_fields),
    }
    write_processed_dataset_manifest(output_root, manifest)
    _emit_progress(
        progress,
        "finished",
        total=len(records),
        new_instance_count=new_count,
        resumed_instance_count=resumed_count,
        skipped_instance_count=skipped_count,
        ignored_count=len(ignored),
        next_plant_number=report["next_plant_number"],
    )
    return report
