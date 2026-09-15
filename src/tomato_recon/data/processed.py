"""Dataset-agnostic persistence and loading for canonical processed plant samples."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from tomato_recon.data.schemas import (
    ORGAN_TYPE_NAMES,
    TOPOLOGY_ROLE_NAMES,
    VISIBILITY_NAMES,
    GraphEdge,
    GraphNode,
    OrganType,
    ParametricPlant,
    PlantGraph,
    PlantSample,
    TopologyRole,
    Visibility,
)


_DATASET_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PLANT_FOLDER = re.compile(r"^plant_(\d+)$")
COMBINED_DATASET = "combined"
POINT_CLOUD_TYPES = ("full", "top_down", "both")


def _plant_number(value: str, *, context: str) -> int:
    match = _PLANT_FOLDER.fullmatch(value)
    if match is None or int(match.group(1)) < 1:
        raise ValueError(f"{context} must be named plant_<positive number>, got {value!r}")
    return int(match.group(1))


def _validate_flat_manifest(manifest: Mapping[str, Any], *, path: Path) -> None:
    if manifest.get("layout") != "flat-plant-instance-v1":
        raise ValueError(
            f"processed dataset manifest does not use the flat plant-instance layout: {path}"
        )
    datasets = manifest.get("datasets")
    instances = manifest.get("instances")
    if not isinstance(datasets, dict) or not isinstance(instances, list):
        raise ValueError(f"invalid processed dataset manifest: {path}")
    for dataset_id, dataset in datasets.items():
        if not _DATASET_ID.fullmatch(str(dataset_id)) or not isinstance(dataset, dict):
            raise ValueError(f"invalid source dataset registry entry {dataset_id!r} in {path}")
        if dataset.get("dataset", dataset_id) != dataset_id:
            raise ValueError(f"source dataset registry key does not match its ID in {path}")

    seen_instance_ids: set[str] = set()
    seen_plant_numbers: set[int] = set()
    seen_sources: set[tuple[str, str]] = set()
    for index, entry in enumerate(instances):
        if not isinstance(entry, dict):
            raise ValueError(f"instance entry {index} is not a mapping in {path}")
        instance_id = str(entry.get("instance_id", ""))
        number = _plant_number(instance_id, context=f"instance entry {index}")
        if instance_id in seen_instance_ids or number in seen_plant_numbers:
            raise ValueError(f"duplicate global plant instance {instance_id!r} in {path}")
        seen_instance_ids.add(instance_id)
        seen_plant_numbers.add(number)
        if "plant_number" in entry and int(entry["plant_number"]) != number:
            raise ValueError(f"plant_number does not match {instance_id!r} in {path}")
        if entry.get("plant_id") != instance_id:
            raise ValueError(
                f"instance {instance_id!r} must use the same globally unique plant_id in {path}"
            )
        if not str(entry.get("source_plant_id", "")).strip():
            raise ValueError(f"instance {instance_id!r} has no source_plant_id in {path}")

        dataset_id = str(entry.get("dataset", ""))
        if dataset_id not in datasets:
            raise ValueError(f"instance {instance_id!r} references unknown dataset {dataset_id!r}")
        source_instance_id = str(entry.get("source_instance_id", "")).strip()
        if not source_instance_id:
            raise ValueError(f"instance {instance_id!r} has no source_instance_id in {path}")
        source_key = (dataset_id, source_instance_id)
        if source_key in seen_sources:
            raise ValueError(f"duplicate source instance {source_key!r} in {path}")
        seen_sources.add(source_key)

        status = str(entry.get("status", ""))
        if status not in {"processing", "complete"}:
            raise ValueError(f"instance {instance_id!r} has invalid status {status!r} in {path}")
        expected_cache = f"{instance_id}/sample.npz"
        if entry.get("cache_file") != expected_cache:
            raise ValueError(
                f"instance {instance_id!r} must use cache_file {expected_cache!r} in {path}"
            )


def processed_dataset_identity(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Return stable, source-agnostic provenance fields from a data config."""
    dataset_id = str(cfg.get("dataset_id", "")).strip()
    if not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError(
            "data.dataset_id must be a lowercase filesystem-safe identifier "
            "containing only letters, digits, underscores, or hyphens"
        )
    dataset_name = str(cfg.get("dataset_name", dataset_id)).strip()
    dataset_version = str(cfg.get("dataset_version", "unknown")).strip()
    if not dataset_name or not dataset_version:
        raise ValueError("data.dataset_name and data.dataset_version must not be empty")
    return {
        "dataset": dataset_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
    }


def validate_instance_id(value: str) -> str:
    """Validate a scan identifier before using it as an instance directory."""
    instance_id = str(value).strip()
    if not _INSTANCE_ID.fullmatch(instance_id):
        raise ValueError(
            "instance_id must be a filesystem-safe identifier containing only letters, "
            "digits, dots, underscores, or hyphens"
        )
    return instance_id


def resolve_processed_dataset_root(cfg: Mapping[str, Any]) -> Path:
    """Validate that every source writes directly into the shared dataset root."""
    processed_dataset_identity(cfg)
    dataset_root_value = str(cfg.get("dataset_root", "")).strip()
    processed_root_value = str(cfg.get("processed_root", "")).strip()
    if not dataset_root_value or not processed_root_value:
        raise ValueError("data.dataset_root and data.processed_root must be configured")
    dataset_root = Path(dataset_root_value).expanduser()
    processed_root = Path(processed_root_value).expanduser()
    if processed_root.resolve() != dataset_root.resolve():
        raise ValueError(
            "data.processed_root must equal data.dataset_root so source datasets share one "
            f"flat plant-instance sequence, got {processed_root} and {dataset_root}"
        )
    return processed_root


def load_processed_dataset_manifest(root: str | Path) -> dict[str, Any]:
    """Load the flat global manifest or create an empty in-memory one."""
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file():
        return {
            "schema_version": "1.0",
            "layout": "flat-plant-instance-v1",
            "datasets": {},
            "instances": [],
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _validate_flat_manifest(manifest, path=path)
    return manifest


def normalise_dataset_selection(dataset: str | None) -> str:
    """Return the canonical source selector used by training and evaluation."""
    selection = COMBINED_DATASET if dataset is None else str(dataset).strip().lower()
    if not selection:
        selection = COMBINED_DATASET
    if not _DATASET_ID.fullmatch(selection):
        raise ValueError(
            "data.dataset must be 'combined' or a lowercase filesystem-safe source "
            "dataset ID"
        )
    return selection


def normalise_point_cloud_type(pcl_type: str | None) -> str:
    """Validate the full/top-down selection used by model training."""
    selection = "full" if pcl_type is None else str(pcl_type).strip().lower()
    if selection not in POINT_CLOUD_TYPES:
        raise ValueError(
            f"data.pcl_type must be one of {POINT_CLOUD_TYPES}, got {pcl_type!r}"
        )
    return selection


def _select_manifest_instances(
    manifest: Mapping[str, Any],
    *,
    split: str | None,
    dataset: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    selection = normalise_dataset_selection(dataset)
    instances = [
        item for item in manifest["instances"] if item.get("status") == "complete"
    ]
    available = sorted({str(item.get("dataset", "")) for item in instances})
    if selection != COMBINED_DATASET:
        if selection not in available:
            raise ValueError(
                f"source dataset {selection!r} is not available; choose one of "
                f"{available or ['<none>']} or {COMBINED_DATASET!r}"
            )
        instances = [item for item in instances if item.get("dataset") == selection]
    if split is not None:
        instances = [
            item
            for item in instances
            if item.get("split", manifest.get("split")) == split
        ]
    return selection, instances


def processed_dataset_compatibility(
    root: str | Path, dataset: str | None = COMBINED_DATASET
) -> dict[str, Any]:
    """Describe all preprocessing variants selected for a training checkpoint.

    The signature deliberately spans every completed split for the selected sources.
    Training, validation, testing, resume, and downstream stages therefore share one
    stable compatibility contract without depending on manifest order or a sampled
    plant instance.
    """
    root = Path(root)
    manifest = load_processed_dataset_manifest(root)
    selection, instances = _select_manifest_instances(
        manifest, split=None, dataset=dataset
    )
    if not instances:
        raise ValueError(f"processed dataset selection {selection!r} contains no samples")

    dataset_hashes: dict[str, list[str]] = {}
    for dataset_id in sorted({str(item["dataset"]) for item in instances}):
        hashes = {
            str(item.get("preprocessing_hash", "")).strip()
            for item in instances
            if item.get("dataset") == dataset_id
        }
        hashes.discard("")
        registry_hash = str(
            manifest.get("datasets", {}).get(dataset_id, {}).get("preprocessing_hash", "")
        ).strip()
        if not hashes and registry_hash:
            hashes.add(registry_hash)
        if not hashes:
            raise ValueError(
                f"source dataset {dataset_id!r} has no preprocessing hash in {root}"
            )
        dataset_hashes[dataset_id] = sorted(hashes)

    contract = {
        "schema_version": "1.0",
        "manifest_schema_version": str(manifest.get("schema_version", "unknown")),
        "layout": str(manifest.get("layout", "unknown")),
        "datasets": dataset_hashes,
    }
    signature = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**contract, "selection": selection, "signature": signature}


def next_plant_number(root: str | Path, manifest: Mapping[str, Any]) -> int:
    """Return one plus the highest plant number present on disk or in the manifest."""
    root = Path(root)
    numbers: list[int] = []
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            numbers.append(
                _plant_number(child.name, context=f"processed dataset directory {child}")
            )
    for entry in manifest.get("instances", []):
        numbers.append(
            _plant_number(
                str(entry.get("instance_id", "")), context="processed manifest instance_id"
            )
        )
    return max(numbers, default=0) + 1


def plant_instance_id(number: int) -> str:
    if number < 1:
        raise ValueError("plant instance number must be positive")
    return f"plant_{number:06d}"


def write_processed_dataset_manifest(root: str | Path, manifest: dict[str, Any]) -> Path:
    """Persist the global manifest with counts derived from completed instances."""
    root = Path(root)
    _validate_flat_manifest(manifest, path=root / "manifest.json")
    completed = [entry for entry in manifest["instances"] if entry.get("status") == "complete"]
    manifest["instance_count"] = len(completed)
    manifest["point_cloud_count"] = len(completed)
    manifest["plant_count"] = len(completed)
    manifest["source_plant_count"] = len(
        {(entry.get("dataset"), entry.get("source_plant_id")) for entry in completed}
    )
    split_counts: dict[str, int] = {}
    for entry in completed:
        split = str(entry.get("split", "unspecified"))
        split_counts[split] = split_counts.get(split, 0) + 1
    manifest["split_counts"] = split_counts
    for dataset_id, dataset in manifest["datasets"].items():
        source_instances = [entry for entry in completed if entry.get("dataset") == dataset_id]
        source_split_counts: dict[str, int] = {}
        for entry in source_instances:
            split = str(entry.get("split", "unspecified"))
            source_split_counts[split] = source_split_counts.get(split, 0) + 1
        dataset["instance_count"] = len(source_instances)
        dataset["point_cloud_count"] = len(source_instances)
        dataset["plant_count"] = len(source_instances)
        dataset["source_plant_count"] = len(
            {entry.get("source_plant_id") for entry in source_instances}
        )
        dataset["split_counts"] = source_split_counts
    root.mkdir(parents=True, exist_ok=True)
    manifest["next_plant_number"] = next_plant_number(root, manifest)
    path = root / "manifest.json"
    temporary_path = root / ".manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
    return path


def save_processed_sample(sample: PlantSample, path: str | Path) -> None:
    """Write one sample using the shared processed-dataset cache contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray("1.0"),
        instance_id=np.asarray(sample.metadata.get("instance_id", sample.plant_id)),
        plant_id=np.asarray(sample.plant_id),
        xyz=sample.xyz.cpu().numpy(),
        rgb=sample.rgb.cpu().numpy(),
        normals=sample.normals.cpu().numpy(),
        semantic=sample.semantic.cpu().numpy(),
        instance=sample.instance.cpu().numpy(),
        point_valid=sample.point_valid.cpu().numpy(),
        node_xyz=sample.node_xyz.cpu().numpy(),
        parent_flow=sample.parent_flow.cpu().numpy(),
        node_valid=sample.node_valid.cpu().numpy(),
        parent_index=sample.parent_index.cpu().numpy(),
        organ_type=sample.organ_type.cpu().numpy(),
        topology_role=sample.topology_role.cpu().numpy(),
        visibility=sample.visibility.cpu().numpy(),
        metadata_json=np.asarray(json.dumps(sample.metadata, sort_keys=True)),
    )


def load_processed_sample(path: str | Path) -> PlantSample:
    """Load one canonical sample independently of its source dataset adapter."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        plant_id = str(cache["plant_id"].item())
        graph_path = path.with_suffix(".graph.json")
        params_path = path.with_suffix(".params.json")
        graph = None
        params = None
        if graph_path.is_file():
            graph = PlantGraph.from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
        if params_path.is_file():
            params = ParametricPlant.from_dict(json.loads(params_path.read_text(encoding="utf-8")))
        metadata = json.loads(str(cache["metadata_json"].item()))
        instance_id = str(
            cache["instance_id"].item()
            if "instance_id" in cache.files
            else metadata.get("instance_id", plant_id)
        )
        metadata.setdefault("instance_id", instance_id)
        pseudo_path = path.parent / "fruit_pseudo.npz"
        metadata["fruit_pseudo_available"] = pseudo_path.is_file()
        metadata["fruit_pseudo_path"] = str(pseudo_path) if pseudo_path.is_file() else None
        sample = PlantSample(
            plant_id=plant_id,
            xyz=torch.from_numpy(cache["xyz"]).float(),
            rgb=torch.from_numpy(cache["rgb"]).float(),
            normals=torch.from_numpy(cache["normals"]).float(),
            semantic=torch.from_numpy(cache["semantic"]).long(),
            instance=torch.from_numpy(cache["instance"]).long(),
            point_valid=torch.from_numpy(cache["point_valid"]).bool(),
            node_xyz=torch.from_numpy(cache["node_xyz"]).float(),
            parent_flow=torch.from_numpy(cache["parent_flow"]).float(),
            node_valid=torch.from_numpy(cache["node_valid"]).bool(),
            parent_index=torch.from_numpy(cache["parent_index"]).long(),
            organ_type=torch.from_numpy(cache["organ_type"]).long(),
            topology_role=torch.from_numpy(cache["topology_role"]).long(),
            visibility=torch.from_numpy(cache["visibility"]).long(),
            graph_target=graph,
            param_target=params,
            metadata=metadata,
        )
    sample.validate()
    return sample


class ProcessedPlantDataset(Dataset[PlantSample]):
    """Load completed point-cloud instances from the shared flat dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        dataset: str | None = COMBINED_DATASET,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"processed manifest not found: {manifest_path}")
        self.manifest = load_processed_dataset_manifest(self.root)
        self.dataset, instances = _select_manifest_instances(
            self.manifest, split=split, dataset=dataset
        )
        self.instances = instances
        self.dataset_ids = tuple(sorted({str(item["dataset"]) for item in instances}))
        cache_root = self.root
        root_resolved = self.root.resolve()
        self.paths = []
        for item in instances:
            path = cache_root / item["cache_file"]
            if not path.resolve().is_relative_to(root_resolved):
                raise ValueError(f"cache path escapes processed dataset root: {path}")
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> PlantSample:
        return load_processed_sample(self.paths[index])


class PointCloudViewDataset(Dataset[PlantSample]):
    """Expose full clouds, top-down clouds, or both for model input."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        dataset: str | None = COMBINED_DATASET,
        pcl_type: str = "full",
    ) -> None:
        self.full_dataset = ProcessedPlantDataset(root, split=split, dataset=dataset)
        self.pcl_type = normalise_point_cloud_type(pcl_type)
        self.variants = ("full", "top_down") if self.pcl_type == "both" else (self.pcl_type,)
        self.dataset = self.full_dataset.dataset
        self.dataset_ids = self.full_dataset.dataset_ids

    def __len__(self) -> int:
        return len(self.full_dataset) * len(self.variants)

    def __getitem__(self, index: int) -> PlantSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index, variant_index = divmod(index, len(self.variants))
        sample = self.full_dataset[source_index]
        variant = self.variants[variant_index]
        if variant == "top_down":
            from tomato_recon.data.top_down import load_top_down_sample

            return load_top_down_sample(self.full_dataset.paths[source_index], sample)
        return replace(sample, metadata={**sample.metadata, "pcl_type": "full"})


def make_tiny_sample(max_nodes: int = 16, num_points: int = 96, seed: int = 7) -> PlantSample:
    """Deterministic topology-rich fixture used by all stage smoke tests."""
    generator = torch.Generator().manual_seed(seed)
    real_nodes = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.05],
            [0.00, 0.00, 0.10],
            [0.00, 0.00, 0.15],
            [0.00, 0.00, 0.20],
            [0.04, 0.00, 0.12],
            [0.08, 0.01, 0.13],
            [-0.04, 0.00, 0.17],
            [-0.08, -0.01, 0.18],
        ],
        dtype=torch.float32,
    )
    parents = torch.tensor([-1, 0, 1, 2, 3, 2, 5, 3, 7], dtype=torch.long)
    if max_nodes < len(real_nodes):
        raise ValueError("tiny fixture requires max_nodes >= 9")
    segments = []
    semantics = []
    for child in range(1, len(real_nodes)):
        parent = int(parents[child])
        t = torch.rand((max(2, num_points // (len(real_nodes) - 1)), 1), generator=generator)
        line = real_nodes[parent] * (1 - t) + real_nodes[child] * t
        jitter = torch.randn(line.shape, generator=generator) * 0.0015
        segments.append(line + jitter)
        semantics.append(
            torch.full(
                (len(line),),
                int(OrganType.MAIN_STEM == OrganType.MAIN_STEM) + (1 if child <= 4 else 3),
                dtype=torch.long,
            )
        )
    xyz = torch.cat(segments)[:num_points]
    semantic = torch.cat(semantics)[:num_points]
    semantic[: min(len(semantic), num_points // 3)] = 2
    if len(xyz) < num_points:
        repeat = num_points - len(xyz)
        xyz = torch.cat([xyz, xyz[:repeat]])
        semantic = torch.cat([semantic, semantic[:repeat]])
    rgb = torch.zeros_like(xyz)
    rgb[:, 1] = 0.55
    normals = torch.nn.functional.normalize(
        torch.randn(xyz.shape, generator=generator), dim=-1
    )
    node_xyz = torch.zeros((max_nodes, 3))
    node_xyz[: len(real_nodes)] = real_nodes
    parent_index = torch.full((max_nodes,), -1, dtype=torch.long)
    parent_index[: len(real_nodes)] = parents
    node_valid = torch.zeros(max_nodes, dtype=torch.bool)
    node_valid[: len(real_nodes)] = True
    parent_flow = torch.zeros_like(node_xyz)
    for child in range(1, len(real_nodes)):
        parent_flow[child] = torch.nn.functional.normalize(
            real_nodes[int(parents[child])] - real_nodes[child], dim=0
        )
    organ_type = torch.full((max_nodes,), int(OrganType.UNKNOWN), dtype=torch.long)
    organ_type[:5] = int(OrganType.MAIN_STEM)
    organ_type[5:] = int(OrganType.UNKNOWN)
    organ_type[5:7] = int(OrganType.SIDE_STEM)
    organ_type[7:9] = int(OrganType.LEAF_STRUCTURE)
    topology = torch.full((max_nodes,), int(TopologyRole.CONTINUATION), dtype=torch.long)
    topology[0] = int(TopologyRole.ROOT)
    topology[2:4] = int(TopologyRole.JUNCTION)
    topology[4] = int(TopologyRole.TIP)
    topology[6] = int(TopologyRole.TIP)
    topology[8] = int(TopologyRole.TIP)
    visibility = torch.full((max_nodes,), int(Visibility.OBSERVED), dtype=torch.long)
    sample = PlantSample(
        plant_id="tiny_tomato",
        xyz=xyz,
        rgb=rgb,
        normals=normals,
        semantic=semantic,
        instance=torch.full((num_points,), -1, dtype=torch.long),
        point_valid=torch.ones(num_points, dtype=torch.bool),
        node_xyz=node_xyz,
        parent_flow=parent_flow,
        node_valid=node_valid,
        parent_index=parent_index,
        organ_type=organ_type,
        topology_role=topology,
        visibility=visibility,
        metadata={
            "schema_version": "1.0",
            "instance_id": "tiny_tomato",
            "preprocessing_hash": "tiny-fixture-v1",
            "normalised_to_original": torch.eye(4).tolist(),
            "dataset": "synthetic-test-fixture",
        },
    )
    nodes = [
        GraphNode(
            id=index,
            xyz=real_nodes[index].tolist(),
            organ_type=ORGAN_TYPE_NAMES[int(organ_type[index])],
            topology_role=TOPOLOGY_ROLE_NAMES[int(topology[index])],
            existence_confidence=1.0,
            visibility=VISIBILITY_NAMES[int(visibility[index])],
            source_slot=index,
        )
        for index in range(len(real_nodes))
    ]
    edges = [
        GraphEdge(
            parent=int(parents[child]),
            child=child,
            edge_type="continuation"
            if organ_type[int(parents[child])] == organ_type[child]
            else "attachment",
            confidence=1.0,
        )
        for child in range(1, len(real_nodes))
    ]
    sample.graph_target = PlantGraph(
        plant_id=sample.plant_id,
        root_node_id=0,
        nodes=nodes,
        edges=edges,
        source={"dataset": "synthetic-test-fixture", "preprocessing_hash": "tiny-fixture-v1"},
    )
    sample.validate()
    return sample
