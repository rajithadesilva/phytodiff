"""Render seeds, flow states and trees against validation/test targets with Pillow."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tomato_recon.data.aggfm import dataset_compatibility, from_sample, sample_points
from tomato_recon.data.processed import PointCloudViewDataset
from tomato_recon.models.flow_matching.runtime import check_evaluation_compatibility, load_runtime, predict


def render(path, data, seeds, prediction, states, frames):
    """Fixed-scale X-Z panels; colours show seed provenance or existence probability."""
    panel_size, margin, header = 500, 25, 50
    selected = sorted(set(np.linspace(0, len(states) - 1, frames, dtype=int).tolist()))
    cloud = data.xyz.cpu().numpy()
    cloud = cloud[np.linspace(0, len(cloud)-1, min(2000, len(cloud)), dtype=int)]
    gt, pred = data.target_xyz.cpu().numpy(), prediction.node_xyz.cpu().numpy()
    extent = np.concatenate((cloud, gt, pred))[:, [0, 2]]
    minimum, maximum = extent.min(0), extent.max(0)
    scale = (panel_size - 2 * margin) / max(float((maximum - minimum).max()), 0.04)
    centre = (minimum + maximum) / 2

    def project(xyz):
        xy = (xyz[:, [0, 2]] - centre) * scale + panel_size / 2
        xy[:, 1] = panel_size - xy[:, 1]
        return xy

    observed = seeds.observed.cpu().numpy()
    initial = seeds.state[:, :3].cpu().numpy()
    panels = [("Observed (orange), latent (blue) seeds", initial, [],
               [(220, 110, 35) if yes else (40, 110, 210) for yes in observed])]
    for index in selected:
        time, state = states[index]
        probability = state[:, 3].sigmoid().numpy()
        colors = [(int(220 * p), int(70 + 130 * p), int(220 * (1-p))) for p in probability]
        panels.append((f"Flow t={time:.2f} | colour = existence", state[:, :3].numpy(), [], colors))
    panels.append((f"Final active nodes + predicted edges ({len(pred)})", pred,
                   prediction.edges.cpu().tolist(), [(220, 110, 35)] * len(pred)))
    target_edges = [(parent, child) for child, parent in enumerate(data.target_parent.cpu().tolist()) if parent >= 0]
    panels.append(("Full GT skeleton (evaluation only)", gt, target_edges, [(185, 42, 34)] * len(gt)))
    columns = min(4, len(panels))
    rows = (len(panels) + columns - 1) // columns
    image = Image.new("RGB", (columns * panel_size, rows * (panel_size + header) + header), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((margin, 15), f"AG-GFM | {data.plant_id} | {data.dataset} | {data.view} | X-Z (metres)",
              fill=(25, 32, 37), font=font)
    for index, (title, xyz, edges, colors) in enumerate(panels):
        panel = Image.new("RGB", (panel_size, panel_size + header), "white")
        painter = ImageDraw.Draw(panel)
        painter.text((12, 12), title, fill=(25, 32, 37), font=font)
        for x, y in project(cloud):
            painter.point((x, y + header), fill=(205, 210, 212))
        pixels = project(xyz)
        for parent, child in edges:
            a, b = pixels[parent], pixels[child]
            painter.line((a[0], a[1]+header, b[0], b[1]+header), fill=colors[child], width=2)
        for (x, y), color in zip(pixels, colors):
            painter.ellipse((x-3, y+header-3, x+3, y+header+3), fill=color)
        painter.rectangle((0, 0, panel_size-1, panel_size+header-1), outline=(180, 185, 190))
        image.paste(panel, ((index % columns) * panel_size, header + (index // columns) * (panel_size + header)))
    image.save(path)


def visualize(checkpoint, output, *, split="test", dataset=None, views=None, processed_root=None,
              steps=None, frames=5, count=1):
    model, encoder, cfg, payload, device = load_runtime(checkpoint)
    root, selection = processed_root or cfg.data.processed_root, dataset or cfg.data.dataset
    check_evaluation_compatibility(dataset_compatibility(root, str(selection)), payload["dataset_compatibility"])
    samples = PointCloudViewDataset(root, split=split, dataset=selection, pcl_types=tuple(views or cfg.data.pcl_types))
    if count < 0 or frames < 2:
        raise ValueError("count must be nonnegative and frames at least two")
    if not len(samples):
        raise ValueError("selected visualization split/views contain no samples")
    selected_count = len(samples.full_dataset) if count == 0 else min(count, len(samples.full_dataset))
    for index in range(selected_count * len(samples.pcl_types)):
        data = sample_points(from_sample(samples[index]), int(cfg.model.aggfm.max_points), int(cfg.seed)).to(device)
        states = []
        prediction, seeds = predict(model, encoder, data.points_only(), cfg, steps=steps,
                                     on_step=lambda i, t, z: states.append((t, z.cpu())))
        folder = Path(output) / split / data.view / data.plant_id
        folder.mkdir(parents=True, exist_ok=True)
        render(folder / "flow.png", data, seeds, prediction, states, frames)
        np.savez_compressed(folder / "trajectory.npz", times=[t for t, _ in states],
                             states=np.stack([z.numpy() for _, z in states]),
                             observed=seeds.observed.cpu().numpy(), active_slots=prediction.source_slots.cpu().numpy())
        (folder / "graph.json").write_text(json.dumps(prediction.graph.to_dict(), indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--dataset", choices=("combined", "tomatowur", "tomatopgt", "pheno4d"))
    parser.add_argument("--pcl-types", nargs="+", choices=("full", "top_down", "side"))
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args(argv)
    visualize(args.checkpoint, args.output, split=args.split, dataset=args.dataset, views=args.pcl_types,
               processed_root=args.processed_root, steps=args.steps, frames=args.frames, count=args.count)


if __name__ == "__main__":
    main()
