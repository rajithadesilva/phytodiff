"""Deterministic parallel-ray visibility and derived fixed-view artifacts."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from tomato_recon.data.schemas import PlantSample

ALGORITHM_VERSION = "xy-footprint-v1"
POINT_FIELDS = ("xyz", "rgb", "normals", "semantic", "instance", "point_valid")


@dataclass(frozen=True)
class FixedViewSpec:
    """Geometry and artifact contract for one parallel-ray sensor."""

    key: str
    filename: str
    algorithm_version: str
    footprint_axes: tuple[int, int]
    depth_axis: int
    view_direction: tuple[float, float, float]
    metadata: Mapping[str, Any]


TOP_DOWN_SPEC = FixedViewSpec(
    key="top_down",
    filename="top_down.npz",
    algorithm_version=ALGORITHM_VERSION,
    footprint_axes=(0, 1),
    depth_axis=2,
    view_direction=(0.0, 0.0, -1.0),
    metadata={"coordinate_frame": {"up_axis": "Z", "meters_per_unit": 1.0}},
)


@dataclass(frozen=True)
class TopDownSettings:
    enabled: bool = True
    occlusion_radius_m: float = 0.001
    depth_tolerance_m: float = 0.001

    def __post_init__(self) -> None:
        validate_view_settings(self, "top_down")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> TopDownSettings:
        return cls(**dict(cfg.get("top_down", {})))


def validate_view_settings(settings: Any, key: str) -> None:
    """Validate settings shared by fixed-view artifact generators."""
    if not isinstance(settings.enabled, bool):
        raise ValueError(f"{key}.enabled must be a boolean")
    for name in ("occlusion_radius_m", "depth_tolerance_m"):
        value = getattr(settings, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_view_indices(
    xyz: np.ndarray,
    point_valid: np.ndarray | None = None,
    *,
    footprint_axes: tuple[int, int],
    depth_axis: int,
    occlusion_radius_m: float = 0.001,
    depth_tolerance_m: float = 0.001,
) -> np.ndarray:
    """Return source rows visible from the positive side of ``depth_axis``.

    Every valid input point contributes a circular footprint in the selected
    projection plane. A point is hidden when another point covering its centre
    is closer to the sensor by more than the depth tolerance. Returned rows
    preserve source order.
    """
    if not np.isfinite(occlusion_radius_m) or occlusion_radius_m < 0:
        raise ValueError("occlusion_radius_m must be finite and non-negative")
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0:
        raise ValueError("depth_tolerance_m must be finite and non-negative")
    axes = (*footprint_axes, depth_axis)
    if len(set(axes)) != 3 or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("footprint_axes and depth_axis must select distinct XYZ axes")
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz must be a finite array with shape [N, 3]")
    valid = np.ones(len(xyz), dtype=bool) if point_valid is None else np.asarray(point_valid)
    if valid.shape != (len(xyz),) or valid.dtype != np.bool_:
        raise ValueError("point_valid must be a boolean array with shape [N]")
    rows = np.flatnonzero(valid)
    if not len(rows):
        raise ValueError("point cloud must contain at least one valid point")
    if occlusion_radius_m == 0:
        return rows

    points = xyz[rows].astype(np.float64)
    footprint = points[:, footprint_axes]
    tree = cKDTree(footprint)
    visible = np.ones(len(rows), dtype=bool)
    # Count first so dense footprints cannot create an unbounded batch of lists.
    for start in range(0, len(rows), 256):
        stop = min(start + 256, len(rows))
        counts = tree.query_ball_point(
            footprint[start:stop], occlusion_radius_m, return_length=True
        )
        cursor = start
        while cursor < stop:
            cumulative = np.cumsum(counts[cursor - start :])
            size = max(1, int(np.searchsorted(cumulative, 262144, side="right")))
            end = min(cursor + size, stop)
            neighbours = tree.query_ball_point(footprint[cursor:end], occlusion_radius_m)
            for index, nearby in enumerate(neighbours, start=cursor):
                visible[index] = (
                    np.max(points[nearby, depth_axis]) - points[index, depth_axis]
                    <= depth_tolerance_m
                )
            cursor = end
    return rows[visible]


def top_down_indices(
    xyz: np.ndarray,
    point_valid: np.ndarray | None = None,
    *,
    occlusion_radius_m: float = 0.001,
    depth_tolerance_m: float = 0.001,
) -> np.ndarray:
    """Return source rows visible along -Z, in their original order."""
    return fixed_view_indices(
        xyz,
        point_valid,
        footprint_axes=TOP_DOWN_SPEC.footprint_axes,
        depth_axis=TOP_DOWN_SPEC.depth_axis,
        occlusion_radius_m=occlusion_radius_m,
        depth_tolerance_m=depth_tolerance_m,
    )


def ensure_fixed_view(
    root: Path,
    entry: dict[str, Any],
    settings: Any,
    spec: FixedViewSpec,
) -> str:
    """Create or repair one fixed-view artifact and update its manifest entry."""
    if not settings.enabled:
        return "disabled"
    source_path = root / entry["cache_file"]
    output_path = source_path.with_name(spec.filename)
    source_hash = file_sha256(source_path)
    contract = {
        "cache_file": output_path.relative_to(root).as_posix(),
        "source_cache_file": entry["cache_file"],
        "source_cache_sha256": source_hash,
        "algorithm_version": spec.algorithm_version,
        "settings": asdict(settings),
    }
    previous = entry.get(spec.key, {})
    if (
        all(previous.get(key) == value for key, value in contract.items())
        and output_path.is_file()
        and previous.get("cache_sha256") == file_sha256(output_path)
    ):
        return "skipped"

    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in POINT_FIELDS}
        count = len(arrays["xyz"])
        for name, values in arrays.items():
            expected = (count, 3) if name in {"xyz", "rgb", "normals"} else (count,)
            if values.shape != expected:
                raise ValueError(f"{source_path}: {name} must have shape {expected}")
        indices = fixed_view_indices(
            arrays["xyz"],
            arrays["point_valid"],
            footprint_axes=spec.footprint_axes,
            depth_axis=spec.depth_axis,
            occlusion_radius_m=settings.occlusion_radius_m,
            depth_tolerance_m=settings.depth_tolerance_m,
        )
        metadata = json.loads(str(source["metadata_json"].item()))
        if "point_to_original_index" in metadata:
            mapping = np.asarray(metadata["point_to_original_index"])
            if mapping.shape != (count,):
                raise ValueError(f"{source_path}: invalid point_to_original_index shape")
            metadata["point_to_original_index"] = mapping[indices].tolist()
        identity = {name: source[name] for name in ("plant_id", "instance_id")}

    valid_count = int(arrays["point_valid"].sum())
    stats = {
        "full_point_count": count,
        "valid_point_count": valid_count,
        "point_count": len(indices),
        "retained_fraction": len(indices) / valid_count,
    }
    metadata[spec.key] = {
        **contract,
        **stats,
        "view_direction": list(spec.view_direction),
        **dict(spec.metadata),
        "full_sample_file": source_path.name,
        "graph_target_file": source_path.with_suffix(".graph.json").name,
        "param_target_file": source_path.with_suffix(".params.json").name,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                schema_version=np.asarray("1.0"),
                **identity,
                **{name: values[indices] for name, values in arrays.items()},
                source_point_indices=indices,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        temporary_path.chmod(source_path.stat().st_mode & 0o666)
        output_hash = file_sha256(temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    entry[spec.key] = {**contract, **stats, "cache_sha256": output_hash}
    return "generated"


def ensure_top_down(
    root: Path, entry: dict[str, Any], settings: TopDownSettings
) -> str:
    """Create or repair one top-down artifact and update its manifest entry."""
    return ensure_fixed_view(root, entry, settings, TOP_DOWN_SPEC)


def load_fixed_view_sample(
    source_path: Path,
    sample: PlantSample,
    spec: FixedViewSpec,
    *,
    generate_command: str,
) -> PlantSample:
    """Pair a fixed-view cloud with the full sample's reconstruction targets."""
    import torch

    path = source_path.with_name(spec.filename)
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run {generate_command} first")
    with np.load(path, allow_pickle=False) as cloud:
        metadata = json.loads(cloud["metadata_json"].item())
        view_metadata = metadata.get(spec.key, {})
        if view_metadata.get("algorithm_version") != spec.algorithm_version:
            raise ValueError(f"{path} uses an old algorithm; rerun {generate_command}")
        if view_metadata.get("source_cache_sha256") != file_sha256(source_path):
            raise ValueError(f"{path} is stale; rerun {generate_command}")
        if str(cloud["plant_id"].item()) != sample.plant_id:
            raise ValueError(f"{path} belongs to a different plant")
        indices = cloud["source_point_indices"]
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) == 0
            or indices.min() < 0
            or indices.max() >= len(sample.xyz)
            or not np.all(np.diff(indices) > 0)
        ):
            raise ValueError(f"{path} has invalid source point indices")
        rows = torch.from_numpy(indices.astype(np.int64))
        values = {name: torch.from_numpy(cloud[name]) for name in POINT_FIELDS}
        for name, value in values.items():
            if not torch.equal(value, getattr(sample, name)[rows]):
                raise ValueError(f"{path}: {name} does not match the full cloud; regenerate it")
        if not bool(values["point_valid"].all()):
            raise ValueError(f"{path} contains invalid source points")
    partial = replace(
        sample,
        **values,
        metadata={
            **sample.metadata,
            **metadata,
            "pcl_type": spec.key,
            "source_point_indices": rows.tolist(),
        },
    )
    partial.validate()
    return partial


def load_top_down_sample(source_path: Path, sample: PlantSample) -> PlantSample:
    """Pair saved top-down input points with the full reconstruction targets."""
    return load_fixed_view_sample(
        source_path, sample, TOP_DOWN_SPEC, generate_command="make generate-top-down"
    )


def generate_fixed_view_dataset(
    root: str | Path,
    settings: Any,
    spec: FixedViewSpec,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Backfill completed manifest instances, preserving full-cloud contracts."""
    from tomato_recon.data.processed import (
        load_processed_dataset_manifest,
        write_processed_dataset_manifest,
    )

    root = Path(root)
    if not (root / "manifest.json").is_file():
        raise ValueError(f"processed dataset manifest is missing: {root / 'manifest.json'}")
    manifest = load_processed_dataset_manifest(root)
    entries = [entry for entry in manifest["instances"] if entry["status"] == "complete"]
    report: dict[str, Any] = {
        "total": len(entries),
        "generated": 0,
        "skipped": 0,
        "disabled": 0,
        "failures": [],
    }
    for current, entry in enumerate(entries, start=1):
        try:
            action = ensure_fixed_view(root, entry, settings, spec)
            if action == "generated":
                write_processed_dataset_manifest(root, manifest)
            report[action] += 1
            update = {"action": action, **entry.get(spec.key, {})}
        except Exception as exc:
            failure = {"instance_id": entry["instance_id"], "error": str(exc)}
            report["failures"].append(failure)
            update = {"action": "failed", **failure}
        if progress is not None:
            progress(
                {
                    "current": current,
                    "total": len(entries),
                    "instance_id": entry["instance_id"],
                    **update,
                }
            )
    return report


def generate_top_down_dataset(
    root: str | Path,
    settings: TopDownSettings | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Backfill top-down artifacts for completed manifest instances."""
    return generate_fixed_view_dataset(
        root, settings or TopDownSettings(), TOP_DOWN_SPEC, progress=progress
    )
