#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from tomato_recon.data.tomatowur import ProcessedTomatoDataset


def render(sample, path: Path) -> None:
    size = 640
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    xyz = sample.xyz[:, [0, 2]]
    minimum, maximum = xyz.amin(0), xyz.amax(0)
    scale = (size - 40) / float((maximum - minimum).max().clamp_min(1e-6))
    pixels = (xyz - minimum) * scale + 20
    colours = (sample.rgb.clamp(0, 1) * 255).byte()
    for point, colour in zip(pixels, colours, strict=True):
        x, y = float(point[0]), size - float(point[1])
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=tuple(colour.tolist()))
    valid = sample.node_valid
    nodes = sample.node_xyz[valid][:, [0, 2]]
    node_pixels = (nodes - minimum) * scale + 20
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    remap = {old: new for new, old in enumerate(valid_indices.tolist())}
    for old in valid_indices.tolist():
        parent = int(sample.parent_index[old])
        if parent >= 0:
            a, b = node_pixels[remap[parent]], node_pixels[remap[old]]
            draw.line((float(a[0]), size - float(a[1]), float(b[0]), size - float(b[1])), fill=(220, 30, 20), width=3)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_preview"))
    args = parser.parse_args()
    dataset = ProcessedTomatoDataset(args.processed_root)
    if len(dataset) < args.count:
        raise ValueError(f"visualisation smoke test needs {args.count} plants, found {len(dataset)}")
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        sample = dataset[index]
        render(sample, args.output / f"{sample.plant_id}.png")
    print(f"wrote {args.count} previews to {args.output}")


if __name__ == "__main__":
    main()
