#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from tomato_recon.data.processed import ProcessedPlantDataset


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


def render(sample, path: Path) -> None:
    panel_size = 520
    header = 52
    margin = 24
    canvas = Image.new("RGB", (panel_size * len(PROJECTIONS), panel_size + header), "white")
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
    point_semantic = sample.semantic[::point_step]
    if bool(sample.metadata.get("rgb_available", True)):
        point_colours = (sample.rgb[::point_step].clamp(0, 1) * 255).byte()
    else:
        point_colours = torch.full((len(point_xyz), 3), 185, dtype=torch.uint8)
        for semantic, colour in SEMANTIC_COLOURS.items():
            point_colours[point_semantic == semantic] = torch.tensor(colour, dtype=torch.uint8)
    valid_indices = sample.node_valid.nonzero(as_tuple=False).flatten()
    nodes = sample.node_xyz[valid_indices]
    remap = {old: new for new, old in enumerate(valid_indices.tolist())}
    foreground = (point_semantic != 0) & (point_semantic != -100)
    bounds_points = point_xyz[foreground] if bool(foreground.any()) else point_xyz
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
        render(sample, args.output / f"{sample.plant_id}.png")
    print(f"wrote {count} canonical dataset previews to {args.output}")


if __name__ == "__main__":
    main()
