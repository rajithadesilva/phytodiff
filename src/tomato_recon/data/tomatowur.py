"""Non-interactive raw-data adapter for TomatoWUR v3."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tomato_recon.data.processed import validate_instance_id
from tomato_recon.data.schemas import IGNORE_INDEX


@dataclass(frozen=True)
class TomatoWURRecord:
    instance_id: str
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
        return [
            {str(k).strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


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
            instance_id = validate_instance_id(str(entry.get("instance_id") or point_path.stem))
            plant_id = str(entry.get("plant_id") or instance_id)
            records.append(
                TomatoWURRecord(
                    instance_id=instance_id,
                    plant_id=plant_id,
                    point_cloud_path=point_path,
                    labels_path=_resolve_relative(base, labels),
                    skeleton_path=_resolve_relative(base, skeleton),
                    genotype=entry.get("genotype") or entry.get("cultivar"),
                    raw_entry=entry,
                )
            )
        self.records = sorted(records, key=lambda item: (item.plant_id, item.instance_id))

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
                raise ValueError(
                    f"skeleton parent {parent_id} does not exist in {record.skeleton_path}"
                )
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
            "instance_id": record.instance_id,
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
