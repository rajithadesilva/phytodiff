#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from tomato_recon.data.tomatowur import ProcessedTomatoDataset


PROJECTIONS = (("front X-Z", (0, 2)), ("side Y-Z", (1, 2)), ("top X-Y", (0, 1)))
POINT_LIMIT = 50_000


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


def render(sample, path: Path, quality: dict[str, Any] | None = None) -> None:
    panel_size = 520
    header = 52
    margin = 24
    canvas = Image.new("RGB", (panel_size * len(PROJECTIONS), panel_size + header), "white")
    draw = ImageDraw.Draw(canvas)
    quality = quality or {"status": "unavailable", "edges": []}
    suspicious = [edge for edge in quality.get("edges", []) if edge.get("suspicious")]
    repairs = quality.get("repairs", [])
    status = str(quality.get("status", "unavailable"))
    status_colour = (180, 20, 120) if status == "review" else (20, 120, 80)
    draw.text((12, 8), f"{sample.plant_id} | skeleton quality: {status}", fill=status_colour)
    draw.text(
        (12, 27),
        "processed: red | suspicious: magenta | repaired replacement: cyan | parent_id→child_id",
        fill=(50, 50, 50),
    )

    point_step = max(1, math.ceil(len(sample.xyz) / POINT_LIMIT))
    point_xyz = sample.xyz[::point_step]
    point_colours = (sample.rgb[::point_step].clamp(0, 1) * 255).byte()
    valid_indices = sample.node_valid.nonzero(as_tuple=False).flatten()
    nodes = sample.node_xyz[valid_indices]
    remap = {old: new for new, old in enumerate(valid_indices.tolist())}
    suspicious_xyz = []
    for edge in suspicious:
        suspicious_xyz.extend([edge["start_xyz"], edge["end_xyz"]])
    for repair in repairs:
        suspicious_xyz.extend([repair["new_start_xyz"], repair["end_xyz"]])
    extra = torch.as_tensor(suspicious_xyz, dtype=sample.xyz.dtype)
    bounds_xyz = (
        torch.cat([point_xyz, nodes, extra], dim=0)
        if len(extra)
        else torch.cat([point_xyz, nodes], dim=0)
    )

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
        for point, colour in zip(pixels, point_colours, strict=True):
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

        for edge in suspicious:
            endpoints = torch.tensor(
                [edge["start_xyz"], edge["end_xyz"]], dtype=sample.xyz.dtype
            )
            edge_pixels = _project(
                endpoints,
                axes,
                minimum,
                scale,
                left=left,
                top=top,
                panel_size=panel_size,
                margin=margin,
            )
            a, b = edge_pixels[0], edge_pixels[1]
            draw.line(
                (float(a[0]), float(a[1]), float(b[0]), float(b[1])),
                fill=(220, 20, 150),
                width=4,
            )
            for point in (a, b):
                x, y = float(point[0]), float(point[1])
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(220, 20, 150))
            midpoint = (a + b) / 2
            edge_label = f"{edge['parent_id']}→{edge['child_id']}"
            draw.text(
                (float(midpoint[0]) + 3, float(midpoint[1]) + 3),
                edge_label,
                fill=(120, 0, 80),
            )

        for repair in repairs:
            endpoints = torch.tensor(
                [repair["new_start_xyz"], repair["end_xyz"]], dtype=sample.xyz.dtype
            )
            edge_pixels = _project(
                endpoints,
                axes,
                minimum,
                scale,
                left=left,
                top=top,
                panel_size=panel_size,
                margin=margin,
            )
            a, b = edge_pixels[0], edge_pixels[1]
            draw.line(
                (float(a[0]), float(a[1]), float(b[0]), float(b[1])),
                fill=(0, 170, 190),
                width=4,
            )
            edge_label = f"{repair['new_parent_id']}→{repair['child_id']}"
            midpoint = (a + b) / 2
            draw.text(
                (float(midpoint[0]) + 3, float(midpoint[1]) + 3),
                edge_label,
                fill=(0, 90, 110),
            )

    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_preview"))
    args = parser.parse_args()
    dataset = ProcessedTomatoDataset(args.processed_root)
    if len(dataset) < args.count:
        raise ValueError(
            f"visualisation smoke test needs {args.count} plants, found {len(dataset)}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    review_count = 0
    repaired_count = 0
    for index in range(args.count):
        sample = dataset[index]
        quality_path = args.processed_root / "samples" / f"{sample.plant_id}.quality.json"
        quality = (
            json.loads(quality_path.read_text(encoding="utf-8"))
            if quality_path.is_file()
            else None
        )
        review_count += int(quality is not None and quality.get("status") == "review")
        repaired_count += int(quality is not None and quality.get("status") == "repaired")
        render(sample, args.output / f"{sample.plant_id}.png", quality)
    print(
        f"wrote {args.count} previews to {args.output}; "
        f"{review_count} require review and {repaired_count} were repaired"
    )


if __name__ == "__main__":
    main()
