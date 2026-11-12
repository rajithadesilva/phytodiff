#!/usr/bin/env python3
"""Render a short DiffTS denoising sequence beside each test plant's GT skeleton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from tomato_recon.data.diffts import from_sample, sample_points
from tomato_recon.data.processed import (
    POINT_CLOUD_TYPES, PointCloudViewDataset, processed_dataset_compatibility,
)
from tomato_recon.models.diffusion.diffts_runtime import (
    create_model, create_stage1_backbone, file_hash, load_stage2_checkpoint, require_stage2_schema, stage1_features,
)
from tomato_recon.models.diffusion.diffts_sampling import create_scheduler, direction_mst, sample


PANEL_SIZE = 640
PANEL_MARGIN = 28
PANEL_GAP = 20
HEADER_HEIGHT = 100
FOOTER_HEIGHT = 55
NODE_RADIUS = 3
EDGE_WIDTH = 3
SCAN_COLOUR = (205, 210, 212)
PREDICTED_COLOUR = (219, 103, 37)
GT_COLOUR = (185, 42, 34)


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _bounds(scan: np.ndarray, target: np.ndarray):
    """Use the same scan and GT extent as Stage 1, independent of diffusion outliers."""
    front = np.concatenate((scan[:, [0, 2]], target[:, [0, 2]]))
    if not np.isfinite(front).all():
        raise ValueError("input scan or ground truth contains non-finite coordinates")
    return front.min(axis=0), front.max(axis=0)


def _project(xyz: np.ndarray, bounds) -> np.ndarray:
    minimum, maximum = bounds
    span = maximum - minimum
    scale = (PANEL_SIZE - 2 * PANEL_MARGIN) / max(float(span.max()), 1e-6)
    projected = (xyz[:, [0, 2]] - minimum) * scale
    x_offset = (PANEL_SIZE - float(span[0]) * scale) / 2
    y_offset = (PANEL_SIZE - float(span[1]) * scale) / 2
    return np.column_stack((
        projected[:, 0] + x_offset,
        PANEL_SIZE - y_offset - projected[:, 1],
    ))


def _draw_skeleton(draw: ImageDraw.ImageDraw, points: np.ndarray, edges: np.ndarray,
                   colour: tuple[int, int, int]) -> int:
    inside = ((points >= 0) & (points < PANEL_SIZE)).all(axis=1)
    for parent, child in edges:
        if inside[parent] and inside[child]:
            draw.line((*points[parent], *points[child]), fill=colour, width=EDGE_WIDTH)
    for x, y in points[inside]:
        draw.ellipse((x - NODE_RADIUS, y - NODE_RADIUS,
                      x + NODE_RADIUS, y + NODE_RADIUS), fill=colour)
    return int((~inside).sum())


def render_frame(path: Path, *, plant_id: str, variant: str, view: str,
                 frame_index: int, frame_count: int, step_index: int, steps: int,
                 timestep: int, scan: np.ndarray,
                 predicted_xyz: np.ndarray, predicted_edges: np.ndarray,
                 target_xyz: np.ndarray, target_edges: np.ndarray, bounds) -> None:
    """Draw large paired front X-Z skeleton views at a fixed scale for this plant."""
    width = 2 * PANEL_SIZE + 3 * PANEL_GAP
    height = HEADER_HEIGHT + PANEL_SIZE + FOOTER_HEIGHT
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(22)
    detail_font = _font(17)
    panel_font = _font(18)
    draw.text((PANEL_GAP, 12), f"Stage 2 diffusion | {plant_id} | {variant} | {view} | front X-Z",
              fill=(25, 32, 37), font=title_font)
    draw.text((PANEL_GAP, 47),
              f"frame {frame_index}/{frame_count} | denoising step {step_index}/{steps}"
              f" | diffusion t={timestep} | predicted {len(predicted_xyz)} nodes"
              f" | GT {len(target_xyz)} nodes",
              fill=(55, 63, 68), font=detail_font)

    predicted_pixels = _project(predicted_xyz, bounds)
    target_pixels = _project(target_xyz, bounds)
    scan_pixels = _project(scan, bounds)
    outside_count = 0
    for column, (name, pixels, edges, colour) in enumerate((
        ("Prediction", predicted_pixels, predicted_edges, PREDICTED_COLOUR),
        ("Ground-truth structure", target_pixels, target_edges, GT_COLOUR),
    )):
        panel = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), "white")
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((0, 0, PANEL_SIZE - 1, PANEL_SIZE - 1),
                             outline=(190, 198, 203), width=1)
        panel_draw.point([tuple(point) for point in scan_pixels], fill=SCAN_COLOUR)
        outside = _draw_skeleton(panel_draw, pixels, edges, colour)
        if column == 0:
            outside_count = outside
        panel_draw.rectangle((5, 5, 286, 37), fill="white")
        panel_draw.text((12, 9), name, fill=(32, 40, 45), font=panel_font)
        image.paste(panel, (PANEL_GAP + column * (PANEL_SIZE + PANEL_GAP), HEADER_HEIGHT))

    legend_y = HEADER_HEIGHT + PANEL_SIZE + 15
    draw.ellipse((PANEL_GAP, legend_y + 3, PANEL_GAP + 8, legend_y + 11),
                 fill=SCAN_COLOUR)
    draw.text((PANEL_GAP + 17, legend_y), "input scan", fill=(55, 63, 68), font=detail_font)
    legend_x = PANEL_GAP + 155
    for label, colour in (("predicted skeleton", PREDICTED_COLOUR),
                          ("ground-truth skeleton", GT_COLOUR)):
        draw.line((legend_x, legend_y + 7, legend_x + 25, legend_y + 7),
                  fill=colour, width=EDGE_WIDTH)
        draw.text((legend_x + 35, legend_y), label, fill=(55, 63, 68), font=detail_font)
        legend_x += 235
    if outside_count:
        draw.text((legend_x, legend_y), f"{outside_count} predicted nodes outside view",
                  fill=(120, 74, 45), font=detail_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _selected_indices(dataset: PointCloudViewDataset, count: int,
                      plant_ids: list[str]) -> list[int]:
    if count < 0:
        raise ValueError("--count must be zero (all plants) or positive")
    instances = dataset.full_dataset.instances
    if plant_ids:
        wanted = set(plant_ids)
        indices = [i for i, entry in enumerate(instances) if entry["plant_id"] in wanted]
        missing = wanted - {instances[i]["plant_id"] for i in indices}
        if missing:
            raise ValueError(f"plant IDs not found in the test split: {sorted(missing)}")
        return indices
    return list(range(len(instances) if count == 0 else min(count, len(instances))))


def _frame_steps(steps: int, frames: int) -> tuple[int, ...]:
    """Select the first and final denoising results plus evenly spaced results between."""
    if steps < 2 or frames < 2 or frames > steps:
        raise ValueError("require 2 <= --frames <= sampling --steps")
    intervals = frames - 1
    return tuple(
        1 + (index * (steps - 1) + intervals // 2) // intervals
        for index in range(frames)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Root directory for plant frame folders")
    parser.add_argument("--view", choices=POINT_CLOUD_TYPES, default="top_down")
    parser.add_argument("--frames", type=int, default=12,
                        help="Evenly spaced output frames, including first and last (default: 12)")
    parser.add_argument("--steps", type=int,
                        help="Denoising steps; defaults to the checkpoint sample_steps")
    parser.add_argument("--count", type=int, default=0,
                        help="First N test plants; 0 selects all (default: 0)")
    parser.add_argument("--plant-id", action="append", default=[],
                        help="Select an exact test plant ID; repeatable, overrides --count")
    parser.add_argument("--max-render-points", type=int, default=2000,
                        help="Maximum input scan points drawn in each frame")
    args = parser.parse_args(argv)
    if args.max_render_points < 1:
        parser.error("--max-render-points must be positive")
    if args.count < 0:
        parser.error("--count must be nonnegative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    require_stage2_schema(checkpoint)
    cfg = OmegaConf.create(checkpoint["config"])
    sampling_steps = (
        int(cfg.model.diffusion.sample_steps) if args.steps is None else args.steps
    )
    if sampling_steps > int(cfg.model.diffusion.train_timesteps):
        parser.error("--steps cannot exceed the checkpoint's training timesteps")
    try:
        frame_steps = _frame_steps(sampling_steps, args.frames)
    except ValueError as exc:
        parser.error(str(exc))
    frame_step_set = set(frame_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DiffTS visualization requires a CUDA GPU")
    compatibility = processed_dataset_compatibility(cfg.data.processed_root, "combined")
    dataset = PointCloudViewDataset(cfg.data.processed_root, split="test", dataset="combined",
                                    pcl_types=(args.view,))
    indices = _selected_indices(dataset, args.count, args.plant_id)
    if not indices:
        raise ValueError("the selected test split has no plants")
    backbone, _ = create_stage1_backbone(cfg, device, load_source=False)
    model = create_model(cfg).to(device).eval()
    load_stage2_checkpoint(args.checkpoint, model, backbone=backbone,
                           compatibility=compatibility, checkpoint=checkpoint)
    if backbone is not None:
        backbone.eval()
    solver = create_scheduler(int(cfg.model.diffusion.train_timesteps),
                              float(cfg.model.diffusion.beta_start),
                              float(cfg.model.diffusion.beta_end))
    output = args.output or Path("outputs/diffts") / model.variant / "test_diffusion_frames"
    run_output = output / args.view / f"frames_{args.frames:03d}_steps_{sampling_steps:03d}"
    run_output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "diffts-stage2-visualization-v3",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "variant": model.variant,
        "split": "test",
        "view": args.view,
        "projection": "front_xz",
        "denoising_steps": sampling_steps,
        "frame_count": args.frames,
        "sampled_steps": list(frame_steps),
        "plants": {},
    }
    with torch.inference_mode():
        for ordinal, index in enumerate(indices, start=1):
            data = sample_points(from_sample(dataset[index]),
                                 int(cfg.model.diffusion.max_points),
                                 seed=int(cfg.seed) + index).to(device)
            point_features, global_features = stage1_features(backbone, data)
            condition = model.encode_condition(data.xyz, data.rgb, data.rgb_available,
                                               point_features, global_features)
            states: list[tuple[int, int, torch.Tensor]] = []
            def capture(step_index: int, timestep: int, state: torch.Tensor) -> None:
                if step_index in frame_step_set:
                    states.append((step_index, timestep, state.detach().cpu()))
            prediction = sample(model, condition, data.xyz, steps=sampling_steps,
                                guidance=float(cfg.model.diffusion.guidance),
                                seed=int(cfg.seed), scheduler=solver, on_step=capture)
            scan = data.xyz.detach().cpu().numpy()
            if len(scan) > args.max_render_points:
                scan = scan[np.linspace(0, len(scan) - 1, args.max_render_points, dtype=int)]
            target_xyz = data.target_xyz.detach().cpu().numpy()
            parents = data.target_parent.detach().cpu().numpy()
            children = np.flatnonzero(parents >= 0)
            target_edges = np.column_stack((parents[children], children))
            bounds = _bounds(data.xyz.detach().cpu().numpy(), target_xyz)
            plant_output = run_output / data.plant_id
            if tuple(step for step, _, _ in states) != frame_steps:
                raise RuntimeError("diffusion sampler did not capture the requested steps")
            frames = []
            for frame_index, (step_index, timestep, state) in enumerate(states, start=1):
                node_xyz = state[:, :3]
                flow = torch.nn.functional.normalize(state[:, 3:], dim=-1)
                _, edges = direction_mst(node_xyz, flow)
                frame_path = plant_output / f"frame_{frame_index:03d}.png"
                render_frame(frame_path, plant_id=data.plant_id, variant=model.variant,
                             view=data.view, frame_index=frame_index,
                             frame_count=args.frames, step_index=step_index,
                             steps=sampling_steps,
                             timestep=timestep, scan=scan,
                             predicted_xyz=node_xyz.numpy(), predicted_edges=edges.numpy(),
                             target_xyz=target_xyz, target_edges=target_edges, bounds=bounds)
                frames.append({"frame": frame_index, "step": step_index,
                               "timestep": timestep,
                               "path": str(frame_path)})
            manifest["plants"][data.plant_id] = {
                "dataset": data.dataset,
                "predicted_count": prediction.predicted_count,
                "ground_truth_count": len(target_xyz),
                "frames": frames,
            }
            (run_output / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[{ordinal}/{len(indices)}] wrote {len(frames)} frames to {plant_output}",
                  flush=True)
    print(f"Stage 2 frames: {run_output}", flush=True)


if __name__ == "__main__":
    main()
