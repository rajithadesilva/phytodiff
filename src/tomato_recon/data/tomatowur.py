"""Non-interactive readers for TomatoWUR v3 and processed caches."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from tomato_recon.data.schemas import (
    ORGAN_TYPE_NAMES,
    TOPOLOGY_ROLE_NAMES,
    VISIBILITY_NAMES,
    GraphEdge,
    GraphNode,
    IGNORE_INDEX,
    OrganType,
    ParametricPlant,
    PlantGraph,
    PlantSample,
    TopologyRole,
    Visibility,
)


@dataclass(frozen=True)
class TomatoWURRecord:
    plant_id: str
    point_cloud_path: Path
    labels_path: Path
    skeleton_path: Path
    genotype: str | None = None
    raw_entry: dict[str, Any] | None = None


def _resolve_relative(base: Path, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else (base / value).resolve()


def _find_key(entry: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        columns: dict[str, list[float]] = {str(name).strip(): [] for name in reader.fieldnames}
        for row in reader:
            for key in columns:
                raw = row.get(key, "")
                try:
                    columns[key].append(float(raw) if raw not in (None, "") else np.nan)
                except ValueError:
                    columns[key].append(np.nan)
    return {key: np.asarray(values) for key, values in columns.items()}


def _read_mixed_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{str(k).strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def _column(
    columns: dict[str, np.ndarray], aliases: tuple[str, ...], length: int, default: float
) -> np.ndarray:
    lower = {key.lower(): key for key in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return columns[lower[alias.lower()]]
    return np.full(length, default)


class TomatoWURReader:
    """Read the public TomatoWUR v3 CSV/JSON layout without downloading it."""

    POINT_KEYS = ("file_name", "point_cloud", "point_cloud_file", "pointcloud_file")
    LABEL_KEYS = ("sem_seg_file_name", "labels", "label_file", "annotation_file")
    SKELETON_KEYS = ("skeleton_file_name", "skeleton", "skeleton_file")

    def __init__(
        self,
        raw_root: str | Path,
        *,
        annotation_version: str = "0-paper-2Dto3D_improved",
        split: str = "train",
        split_file: str | Path | None = None,
    ) -> None:
        self.raw_root = Path(raw_root).expanduser().resolve()
        if not self.raw_root.is_dir():
            raise FileNotFoundError(
                f"TomatoWUR root does not exist: {self.raw_root}. "
                "Mount/download the dataset outside the container; this reader never downloads it."
            )
        if split_file is None:
            candidates = [
                self.raw_root / "ann_versions" / annotation_version / "json" / f"{split}.json",
                self.raw_root / "ann_versions" / annotation_version / "jsons" / f"{split}.json",
                self.raw_root / "splits" / f"{split}.json",
                self.raw_root / f"{split}.json",
            ]
            self.split_file = next((path for path in candidates if path.is_file()), candidates[0])
        else:
            self.split_file = Path(split_file).expanduser().resolve()
        if not self.split_file.is_file():
            raise FileNotFoundError(
                f"TomatoWUR split file not found: {self.split_file}. "
                "Set data.split_file to an existing plant-level split JSON."
            )
        with self.split_file.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        if isinstance(entries, dict):
            entries = entries.get("samples", entries.get("plants", entries.get(split, [])))
        if not isinstance(entries, list):
            raise ValueError(f"split JSON must contain a list of samples: {self.split_file}")
        base = self.split_file.parent
        records: list[TomatoWURRecord] = []
        for entry in entries:
            point = _find_key(entry, self.POINT_KEYS)
            labels = _find_key(entry, self.LABEL_KEYS)
            skeleton = _find_key(entry, self.SKELETON_KEYS)
            if not point or not labels or not skeleton:
                raise ValueError(
                    f"split entry must provide point cloud, labels, and skeleton paths: {entry}"
                )
            point_path = _resolve_relative(base, point)
            plant_id = str(entry.get("plant_id") or point_path.stem)
            records.append(
                TomatoWURRecord(
                    plant_id=plant_id,
                    point_cloud_path=point_path,
                    labels_path=_resolve_relative(base, labels),
                    skeleton_path=_resolve_relative(base, skeleton),
                    genotype=entry.get("genotype") or entry.get("cultivar"),
                    raw_entry=entry,
                )
            )
        self.records = sorted(records, key=lambda item: item.plant_id)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @staticmethod
    def load_record(record: TomatoWURRecord) -> dict[str, Any]:
        missing = [
            str(path)
            for path in (record.point_cloud_path, record.labels_path, record.skeleton_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("TomatoWUR sample files are missing: " + ", ".join(missing))
        pc = _read_numeric_csv(record.point_cloud_path)
        n = len(_column(pc, ("x",), 0, np.nan))
        xyz = np.stack(
            [_column(pc, (axis,), n, np.nan) for axis in ("x", "y", "z")], axis=-1
        ).astype(np.float32)
        # The public files are BGR; expose the canonical RGB contract.
        rgb = np.stack(
            [_column(pc, (channel,), n, 0.0) for channel in ("red", "green", "blue")],
            axis=-1,
        ).astype(np.float32)
        if not np.isfinite(rgb).all() or rgb.min(initial=0) < 0 or rgb.max(initial=0) > 255:
            raise ValueError(f"RGB values must be finite and in [0,255]: {record.point_cloud_path}")
        if rgb.size and np.nanmax(rgb) > 1.0:
            rgb /= 255.0
        normals = np.stack(
            [_column(pc, aliases, n, 0.0) for aliases in (("nx",), ("ny",), ("nz",))],
            axis=-1,
        ).astype(np.float32)
        if not np.isfinite(normals).all():
            raise ValueError(f"surface normals must be finite: {record.point_cloud_path}")
        labels = _read_numeric_csv(record.labels_path)
        if labels and len(next(iter(labels.values()))) != n:
            raise ValueError(
                f"point/label row mismatch for {record.plant_id}: {n} versus "
                f"{len(next(iter(labels.values())))}"
            )
        semantic_values = _column(
            labels,
            ("semantic", "semantic_label", "semantics", "class", "label"),
            n,
            IGNORE_INDEX,
        )
        # TomatoWUR uses the common uint8 sentinel 255 for points without a
        # semantic annotation. Convert it to the project's loss ignore index
        # instead of treating it as an additional semantic class.
        semantic = np.where(
            np.isfinite(semantic_values) & (semantic_values != 255),
            semantic_values,
            IGNORE_INDEX,
        ).astype(np.int64)
        valid_semantic = set(np.unique(semantic).tolist()) - {IGNORE_INDEX}
        if not valid_semantic.issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"unsupported TomatoWUR semantic labels: {sorted(valid_semantic)}")
        instance_values = _column(
            labels,
            ("leaf_stem_instances", "instance", "instance_id", "leaf_instances"),
            n,
            -1,
        )
        instance = np.where(np.isfinite(instance_values), instance_values, -1).astype(np.int64)

        rows = _read_mixed_csv(record.skeleton_path)
        valid_rows: list[dict[str, str]] = []
        for row in rows:
            try:
                coords = [float(row[key]) for key in ("x_skeleton", "y_skeleton", "z_skeleton")]
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(coords).all():
                valid_rows.append(row)
        if not valid_rows:
            raise ValueError(f"skeleton contains no finite nodes: {record.skeleton_path}")
        original_ids = []
        for index, row in enumerate(valid_rows):
            try:
                original_ids.append(int(float(row.get("vid", index))))
            except ValueError:
                original_ids.append(index)
        if len(set(original_ids)) != len(original_ids):
            raise ValueError(f"skeleton vid values must be unique: {record.skeleton_path}")
        expected_ids = set(range(len(valid_rows)))
        if set(original_ids) != expected_ids:
            raise ValueError(
                "official skeleton vid values must cover the coordinate-row indices "
                f"0..{len(valid_rows) - 1}: {record.skeleton_path}"
            )

        # This is an edge-list table, not a node table sorted by vid. As in the
        # upstream TomatoWUR loader, coordinates are indexed by CSV row while
        # each row's vid/parentid fields describe one edge between row indices.
        node_xyz = np.asarray(
            [
                [float(row[key]) for key in ("x_skeleton", "y_skeleton", "z_skeleton")]
                for row in valid_rows
            ],
            dtype=np.float32,
        )
        parent = np.full(len(valid_rows), -1, dtype=np.int64)
        edge_type = np.full(len(valid_rows), "", dtype=object)
        for row in valid_rows:
            try:
                child_id = int(float(row.get("vid", "")))
            except ValueError:
                continue
            raw_parent = row.get("parentid", "")
            try:
                parent_id = int(float(raw_parent))
            except ValueError:
                continue
            if parent_id == child_id:
                continue
            if parent_id not in expected_ids:
                raise ValueError(f"skeleton parent {parent_id} does not exist in {record.skeleton_path}")
            parent[child_id] = parent_id
            edge_type[child_id] = row.get("edgetype", "")
        roots = np.flatnonzero(parent < 0)
        if len(roots) != 1:
            raise ValueError(
                f"official skeleton must have exactly one root, found {len(roots)}: "
                f"{record.skeleton_path}"
            )
        root = int(roots[0])
        traits: dict[str, np.ndarray] = {}
        for name in ("gt_int_length", "gt_int_diameter", "gt_ph_angle", "gt_lf_angle"):
            values = []
            for row in valid_rows:
                try:
                    values.append(float(row.get(name, "")))
                except ValueError:
                    values.append(np.nan)
            traits[name] = np.asarray(values, dtype=np.float32)
        return {
            "plant_id": record.plant_id,
            "xyz": xyz,
            "rgb": np.nan_to_num(rgb),
            "normals": np.nan_to_num(normals),
            "semantic": semantic,
            "instance": instance,
            "node_xyz": node_xyz,
            "node_ids": np.arange(len(valid_rows), dtype=np.int64),
            "parent_index": parent,
            "edge_type": edge_type,
            "root_index": root,
            "traits": traits,
            "genotype": record.genotype,
        }


def save_processed_sample(sample: PlantSample, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray("1.0"),
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
        pseudo_path = path.parents[1] / "fruit_pseudo" / f"{plant_id}.npz"
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


class ProcessedTomatoDataset(Dataset[PlantSample]):
    def __init__(self, root: str | Path, split: str | None = None) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"processed manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = self.manifest.get("samples", [])
        if split is not None:
            samples = [item for item in samples if item.get("split", self.manifest.get("split")) == split]
        self.paths = [self.root / "samples" / item["cache_file"] for item in samples]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> PlantSample:
        return load_processed_sample(self.paths[index])


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
