"""Fixed +Y sensor visibility and side-view point-cloud artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tomato_recon.data.top_down import (
    FixedViewSpec,
    ensure_fixed_view,
    fixed_view_indices,
    generate_fixed_view_dataset,
    load_fixed_view_sample,
    validate_view_settings,
)

if TYPE_CHECKING:
    from tomato_recon.data.schemas import PlantSample

ALGORITHM_VERSION = "xz-footprint-v1"

SIDE_SPEC = FixedViewSpec(
    key="side",
    filename="side.npz",
    algorithm_version=ALGORITHM_VERSION,
    footprint_axes=(0, 2),
    depth_axis=1,
    view_direction=(0.0, -1.0, 0.0),
    metadata={
        "coordinate_frame": {"up_axis": "Z", "meters_per_unit": 1.0},
        "projection_plane": "XZ",
        "projection_axes": ["X", "Z"],
        "sensor_side": "+Y",
    },
)


@dataclass(frozen=True)
class SideSettings:
    enabled: bool = True
    occlusion_radius_m: float = 0.001
    depth_tolerance_m: float = 0.001

    def __post_init__(self) -> None:
        validate_view_settings(self, "side")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> SideSettings:
        return cls(**dict(cfg.get("side", {})))


def side_indices(
    xyz: np.ndarray,
    point_valid: np.ndarray | None = None,
    *,
    occlusion_radius_m: float = 0.001,
    depth_tolerance_m: float = 0.001,
) -> np.ndarray:
    """Return source rows visible from +Y looking along -Y, in source order."""
    return fixed_view_indices(
        xyz,
        point_valid,
        footprint_axes=SIDE_SPEC.footprint_axes,
        depth_axis=SIDE_SPEC.depth_axis,
        occlusion_radius_m=occlusion_radius_m,
        depth_tolerance_m=depth_tolerance_m,
    )


def ensure_side(root: Path, entry: dict[str, Any], settings: SideSettings) -> str:
    """Create or repair one side artifact and update its manifest entry."""
    return ensure_fixed_view(root, entry, settings, SIDE_SPEC)


def load_side_sample(source_path: Path, sample: PlantSample) -> PlantSample:
    """Pair saved side-view input points with the full reconstruction targets."""
    return load_fixed_view_sample(
        source_path, sample, SIDE_SPEC, generate_command="make generate-side"
    )


def generate_side_dataset(
    root: str | Path,
    settings: SideSettings | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Backfill side artifacts for completed manifest instances."""
    return generate_fixed_view_dataset(
        root, settings or SideSettings(), SIDE_SPEC, progress=progress
    )
