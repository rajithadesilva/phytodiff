#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tomato_recon.data.processed import ProcessedPlantDataset
from tomato_recon.data.top_down import file_sha256


PROJECTIONS = (("front X-Z", (0, 2)), ("side Y-Z", (1, 2)), ("top X-Y", (0, 1)))
POINT_LIMIT = 50_000
SEMANTIC_COLOURS = {
    -100: (225, 225, 225),
    0: (215, 215, 215),
    1: (45, 155, 70),
    2: (65, 105, 45),
    3: (70, 70, 70),
    4: (205, 125, 35),
}


@dataclass(frozen=True)
class FixedView:
    source_indices: torch.Tensor
    occlusion_radius_m: float
    depth_tolerance_m: float
    key: str
    view_direction: tuple[float, float, float]


TopDownView = FixedView


def load_fixed_view(source_path: Path, sample, key: str) -> FixedView | None:
    path = source_path.with_name(f"{key}.npz")
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cloud:
        metadata = json.loads(cloud["metadata_json"].item())[key]
        if metadata["source_cache_sha256"] != file_sha256(source_path):
            raise ValueError(f"{path} is stale; rerun scripts/generate_{key}.py")
        indices = cloud["source_point_indices"]
        if (
            indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) == 0 or indices.min() < 0 or indices.max() >= len(sample.xyz)
            or not np.all(np.diff(indices) > 0)
        ):
            raise ValueError(f"{path} has invalid source point indices")
        rows = torch.from_numpy(indices.astype(np.int64))
        if not torch.equal(torch.from_numpy(cloud["xyz"]), sample.xyz[rows]):
            raise ValueError(f"{path} does not match its full cloud; regenerate {key} clouds")
        if not bool(sample.point_valid[rows].all()):
            raise ValueError(f"{path} contains invalid source points")
        return FixedView(
            rows,
            **{
                name: float(metadata["settings"][name])
                for name in ("occlusion_radius_m", "depth_tolerance_m")
            },
            key=key,
            view_direction=tuple(float(value) for value in metadata["view_direction"]),
        )


def load_top_down(source_path: Path, sample) -> FixedView | None:
    return load_fixed_view(source_path, sample, "top_down")


def load_side(source_path: Path, sample) -> FixedView | None:
    return load_fixed_view(source_path, sample, "side")


def _point_colours(sample) -> torch.Tensor:
    if bool(sample.metadata.get("rgb_available", True)):
        return (sample.rgb.clamp(0, 1) * 255).byte()
    colours = torch.full((len(sample.xyz), 3), 185, dtype=torch.uint8)
    for semantic, colour in SEMANTIC_COLOURS.items():
        colours[sample.semantic == semantic] = torch.tensor(colour, dtype=torch.uint8)
    return colours


def _camera_basis(azimuth_deg: float, elevation_deg: float) -> torch.Tensor:
    azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
    return torch.tensor([
        [-math.sin(azimuth), math.cos(azimuth), 0],
        [-math.sin(elevation) * math.cos(azimuth),
         -math.sin(elevation) * math.sin(azimuth), math.cos(elevation)],
        [math.cos(elevation) * math.cos(azimuth),
         math.cos(elevation) * math.sin(azimuth), math.sin(elevation)],
    ], dtype=torch.float32)


def _perspective(
    xyz: torch.Tensor, centre: torch.Tensor, basis: torch.Tensor, distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = (xyz - centre) @ basis.T
    depth = distance - camera[:, 2]
    return camera[:, :2] * distance / depth.clamp_min(1e-6)[:, None], depth


def _draw_perspective_row(
    draw: ImageDraw.ImageDraw, sample, view: FixedView | None,
    bounds_xyz: torch.Tensor, colours: torch.Tensor,
    *, panel_size: int, top: int, azimuth_deg: float, elevation_deg: float,
    view_label: str, view_direction: tuple[float, float, float],
) -> None:
    centre = (bounds_xyz.amin(dim=0) + bounds_xyz.amax(dim=0)) / 2
    radius = float(torch.linalg.vector_norm(bounds_xyz - centre, dim=1).max().clamp_min(1e-5))
    distance = radius * 5  # Distant camera gives a mild perspective from the side.
    basis = _camera_basis(azimuth_deg, elevation_deg)
    bounds, _ = _perspective(bounds_xyz, centre, basis, distance)
    minimum, maximum = bounds.amin(dim=0), bounds.amax(dim=0)
    midpoint = (minimum + maximum) / 2
    scale = (panel_size - 110) / float((maximum - minimum).max().clamp_min(1e-6))
    projected, depths = _perspective(sample.xyz, centre, basis, distance)
    relative = (projected - midpoint) * scale
    relative[:, 0] += panel_size / 2
    relative[:, 1] = panel_size / 2 + 8 - relative[:, 1]
    inside = (
        (depths > 0) & (relative[:, 0] >= 20) & (relative[:, 0] < panel_size - 20)
        & (relative[:, 1] >= 55) & (relative[:, 1] < panel_size - 45)
        & sample.point_valid
    )
    full_rows = sample.point_valid.nonzero(as_tuple=False).flatten()
    retained = torch.zeros(len(sample.xyz), dtype=torch.bool)
    if view is not None:
        retained[view.source_indices] = True

    def points(rows: torch.Tensor, left: int, colour: tuple[int, int, int] | None = None) -> None:
        rows = rows[inside[rows]]
        rows = rows[::max(1, math.ceil(len(rows) / POINT_LIMIT))]
        # Paint distant points first so the nearest surface wins screen overlaps.
        rows = rows[torch.argsort(depths[rows], descending=True, stable=True)]
        sparse = len(rows) < 5000
        for index in rows.tolist():
            x, y = relative[index].tolist()
            fill = colour or tuple(colours[index].tolist())
            if sparse:
                draw.ellipse((left + x - 1, top + y - 1, left + x + 1, top + y + 1), fill=fill)
            else:
                draw.point((left + x, top + y), fill=fill)

    labels = (
        "Full cloud | side perspective",
        f"{view_label} cloud | same camera",
        f"{view_label} occlusion | green retained, grey hidden",
    )
    for panel, label in enumerate(labels):
        left = panel * panel_size
        draw.rectangle((left, top, left + panel_size - 1, top + panel_size - 1),
                       outline=(210, 210, 210))
        draw.text((left + 10, top + 9), label, fill=(20, 20, 20))
        if panel == 0:
            points(full_rows, left)
            detail = f"{len(full_rows):,} valid points | no skeleton overlay"
        elif view is None:
            draw.text((left + 30, top + panel_size // 2),
                      f"{view_label} cloud unavailable. Run generate_{view_label}.py.",
                      fill=(95, 95, 95))
            continue
        else:
            if panel == 2:
                points(full_rows[~retained[full_rows]], left, (180, 187, 197))
            points(view.source_indices, left, (20, 140, 70) if panel == 2 else None)
            fraction = len(view.source_indices) / max(1, len(full_rows))
            detail = (f"{len(view.source_indices):,} points | {fraction:.1%} retained"
                      if panel == 1 else "Hidden points shown as a grey reference")
        draw.text((left + 10, top + panel_size - 24), detail, fill=(65, 65, 65))
        # Project the world-space sensor ray through the same perspective camera.
        ray = torch.tensor(view_direction) * radius * 0.2
        arrow_xyz = torch.stack([centre - ray, centre])
        arrow, _ = _perspective(arrow_xyz, centre, basis, distance)
        direction = arrow[1] - arrow[0]
        direction[1] *= -1
        direction /= torch.linalg.vector_norm(direction).clamp_min(1e-6)
        start = torch.tensor([left + panel_size - 34., top + 42.])
        end = start + direction * 42
        perpendicular = torch.tensor([-direction[1], direction[0]])
        draw.line((*start.tolist(), *end.tolist()), fill=(45, 85, 160), width=2)
        draw.polygon([tuple(end.tolist()), tuple((end - direction * 9 + perpendicular * 4).tolist()),
                      tuple((end - direction * 9 - perpendicular * 4).tolist())], fill=(45, 85, 160))
        sensor_axis = "-Z" if view_label == "top_down" else "-Y"
        draw.text(
            (left + panel_size - 127, top + 29),
            f"sensor {sensor_axis}",
            fill=(45, 85, 160),
        )


def _project(
    values: torch.Tensor,
    axes: tuple[int, int],
    minimum: torch.Tensor,
    scale: float,
    *,
    left: int,
    top: int,
    panel_size: int,
    margin: int,
) -> torch.Tensor:
    projected = (values[:, list(axes)] - minimum) * scale
    x = projected[:, 0] + left + margin
    y = top + panel_size - margin - projected[:, 1]
    return torch.stack([x, y], dim=-1)


def render(
    sample, path: Path, top_down: FixedView | None = None,
    side: FixedView | None = None,
    *, azimuth_deg: float = 35.0, elevation_deg: float = 15.0,
) -> None:
    panel_size = 520
    header = 52
    margin = 24
    if not math.isfinite(azimuth_deg) or not math.isfinite(elevation_deg):
        raise ValueError("camera angles must be finite")
    if not -80 <= elevation_deg <= 80:
        raise ValueError("elevation must be between -80 and 80 degrees for a side view")
    canvas = Image.new(
        "RGB", (panel_size * len(PROJECTIONS), 3 * panel_size + header), "white"
    )
    draw = ImageDraw.Draw(canvas)
    source = sample.metadata.get("skeleton_source", "unknown")
    dataset = sample.metadata.get("dataset", "unknown")
    source_instance = sample.metadata.get("source_instance_id", "unknown")
    draw.text(
        (12, 8),
        f"{sample.plant_id} | {dataset}:{source_instance}",
        fill=(20, 120, 80),
    )
    draw.text(
        (12, 27),
        f"skeleton: {source} | cached target edges: red",
        fill=(50, 50, 50),
    )

    point_step = max(1, math.ceil(len(sample.xyz) / POINT_LIMIT))
    point_xyz = sample.xyz[::point_step]
    colours = _point_colours(sample)
    point_colours = colours[::point_step]
    valid_indices = sample.node_valid.nonzero(as_tuple=False).flatten()
    nodes = sample.node_xyz[valid_indices]
    remap = {old: new for new, old in enumerate(valid_indices.tolist())}
    foreground = (sample.semantic != 0) & (sample.semantic != -100) & sample.point_valid
    bounds_points = sample.xyz[foreground] if bool(foreground.any()) else sample.xyz[sample.point_valid]
    bounds_xyz = torch.cat([bounds_points, nodes], dim=0)

    for panel_index, (label, axes) in enumerate(PROJECTIONS):
        left = panel_index * panel_size
        top = header
        bounds = bounds_xyz[:, list(axes)]
        minimum = bounds.amin(dim=0)
        maximum = bounds.amax(dim=0)
        scale = (panel_size - 2 * margin) / float((maximum - minimum).max().clamp_min(1e-6))
        draw.rectangle(
            (left, top, left + panel_size - 1, top + panel_size - 1),
            outline=(210, 210, 210),
        )
        draw.text((left + 8, top + 7), label, fill=(20, 20, 20))

        pixels = _project(
            point_xyz,
            axes,
            minimum,
            scale,
            left=left,
            top=top,
            panel_size=panel_size,
            margin=margin,
        )
        inside = (
            (pixels[:, 0] >= left)
            & (pixels[:, 0] < left + panel_size)
            & (pixels[:, 1] >= top)
            & (pixels[:, 1] < top + panel_size)
        )
        for point, colour in zip(pixels[inside], point_colours[inside], strict=True):
            x, y = float(point[0]), float(point[1])
            draw.point((x, y), fill=tuple(colour.tolist()))

        node_pixels = _project(
            nodes,
            axes,
            minimum,
            scale,
            left=left,
            top=top,
            panel_size=panel_size,
            margin=margin,
        )
        for old in valid_indices.tolist():
            parent = int(sample.parent_index[old])
            if parent >= 0:
                a, b = node_pixels[remap[parent]], node_pixels[remap[old]]
                draw.line(
                    (float(a[0]), float(a[1]), float(b[0]), float(b[1])),
                    fill=(215, 45, 35),
                    width=1,
                )

    _draw_perspective_row(
        draw, sample, top_down, bounds_xyz, colours, panel_size=panel_size,
        top=header + panel_size, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
        view_label="top_down", view_direction=(0.0, 0.0, -1.0),
    )
    _draw_perspective_row(
        draw, sample, side, bounds_xyz, colours, panel_size=panel_size,
        top=header + 2 * panel_size, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
        view_label="side", view_direction=(0.0, -1.0, 0.0),
    )
    if top_down is not None:
        draw.text((760, 27),
                  f"Top-down: radius {top_down.occlusion_radius_m:g} m, "
                  f"depth tolerance {top_down.depth_tolerance_m:g} m | "
                  f"view az {azimuth_deg:g}, elev {elevation_deg:g} deg", fill=(50, 50, 50))
    if side is not None:
        draw.text((1130, 27),
                  f"Side: radius {side.occlusion_radius_m:g} m, "
                  f"depth tolerance {side.depth_tolerance_m:g} m", fill=(50, 50, 50))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render canonical plant instances from a flat processed dataset"
    )
    parser.add_argument("processed_root", type=Path)
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of completed instances to render; use 0 for the entire dataset",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_preview"))
    parser.add_argument("--azimuth-deg", type=float, default=35.0,
                        help="Azimuth of the shared side-perspective camera (default: 35)")
    parser.add_argument("--elevation-deg", type=float, default=15.0,
                        help="Elevation of the shared side-perspective camera (default: 15)")
    args = parser.parse_args()
    dataset = ProcessedPlantDataset(args.processed_root)
    if args.count < 0:
        raise ValueError("--count must be zero (all instances) or a positive integer")
    count = len(dataset) if args.count == 0 else args.count
    if count == 0:
        raise ValueError(f"processed dataset contains no completed instances: {args.processed_root}")
    if len(dataset) < count:
        raise ValueError(
            f"visualisation requested {count} instances, found {len(dataset)} completed instances"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        sample = dataset[index]
        print(
            f"[{index + 1:>{len(str(count))}}/{count}] "
            f"{sample.plant_id} ({sample.metadata.get('dataset', 'unknown')}): rendering",
            flush=True,
        )
        render(sample, args.output / f"{sample.plant_id}.png",
               load_top_down(dataset.paths[index], sample),
               load_side(dataset.paths[index], sample),
               azimuth_deg=args.azimuth_deg, elevation_deg=args.elevation_deg)
    print(f"wrote {count} canonical dataset previews to {args.output}")


if __name__ == "__main__":
    main()
