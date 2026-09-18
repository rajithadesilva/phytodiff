#!/usr/bin/env python3
"""Render Stage 1 encoder predictions beside their processed targets."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from tomato_recon.data.collate import collate_plant_samples
from tomato_recon.data.schemas import IGNORE_INDEX, EncoderOutput, PlantSample, TopologyRole
from tomato_recon.data.processed import (
    POINT_CLOUD_TYPES,
    ProcessedPlantDataset,
    normalise_dataset_selection,
    normalise_point_cloud_types,
    processed_dataset_compatibility,
)
from tomato_recon.data.side import load_side_sample
from tomato_recon.data.top_down import load_top_down_sample
from tomato_recon.evaluation.encoder import EncoderMetricAccumulator, encoder_metrics_for_sample
from tomato_recon.models.encoders.base import PointEncoder, encoder_losses
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.train.common import checkpoint_sha256, load_checkpoint


SEMANTIC_COLOURS = {
    IGNORE_INDEX: (185, 185, 185),
    0: (90, 90, 90),       # background
    1: (55, 170, 70),      # leaf
    2: (205, 70, 45),      # main stem
    3: (55, 105, 210),     # support pole
    4: (235, 155, 35),     # side stem
}
SEMANTIC_NAMES = {
    0: "background",
    1: "leaf",
    2: "main stem",
    3: "support pole",
    4: "side stem",
}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot access a CUDA GPU")
    return torch.device(value)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "plant"


def _sample_indices(length: int, limit: int) -> torch.Tensor:
    if limit <= 0 or length <= limit:
        return torch.arange(length)
    return torch.linspace(0, length - 1, steps=limit).round().long().unique()


def _probability_colours(probability: torch.Tensor, kind: str) -> torch.Tensor:
    probability = probability.detach().cpu().float().clamp(0, 1)
    if kind == "skeleton":
        # Dark blue -> cyan -> yellow.
        red = 255 * probability.square()
        green = 35 + 220 * probability
        blue = 210 * (1 - probability) + 35
    else:
        # Dark purple -> magenta -> yellow.
        red = 45 + 210 * probability
        green = 25 + 220 * probability.square()
        blue = 120 + 120 * (1 - probability)
    return torch.stack([red, green, blue], dim=-1).byte()


def _semantic_colours(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.detach().cpu().long()
    colours = torch.empty((len(labels), 3), dtype=torch.uint8)
    for label in labels.unique().tolist():
        colours[labels == label] = torch.tensor(
            SEMANTIC_COLOURS.get(int(label), (255, 0, 255)), dtype=torch.uint8
        )
    return colours


def _project(
    values: torch.Tensor,
    minimum: torch.Tensor,
    scale: float,
    *,
    left: int,
    top: int,
    panel_size: int,
    margin: int,
) -> torch.Tensor:
    # Stage 1 uses a front X-Z projection; Z is the canonical up axis.
    projected = (values[:, [0, 2]].cpu() - minimum) * scale
    x = projected[:, 0] + left + margin
    y = top + panel_size - margin - projected[:, 1]
    return torch.stack([x, y], dim=-1)


def _draw_points(draw: ImageDraw.ImageDraw, pixels: torch.Tensor, colours: torch.Tensor) -> None:
    for point, colour in zip(pixels, colours, strict=True):
        draw.point(
            (int(point[0]), int(point[1])),
            fill=tuple(int(channel) for channel in colour),
        )


def encoder_metrics(
    sample: PlantSample,
    output: EncoderOutput,
    *,
    skeleton_threshold_m: float,
    junction_threshold_multiplier: float,
    probability_threshold: float,
) -> dict[str, float]:
    """Compute canonical Stage 1 metrics for one unpadded plant."""
    return encoder_metrics_for_sample(
        sample,
        output,
        skeleton_threshold_m=skeleton_threshold_m,
        junction_threshold_multiplier=junction_threshold_multiplier,
        probability_threshold=probability_threshold,
    )


def render_encoder_prediction(
    sample: PlantSample,
    output: EncoderOutput,
    metrics: dict[str, float],
    path: Path,
    *,
    probability_threshold: float = 0.5,
    max_render_points: int = 50_000,
) -> None:
    """Write the six Stage 1 diagnostic panels for one plant."""
    titles = (
        "1  Input RGB",
        "2  Ground-truth semantics",
        "3  Predicted semantics",
        "4  Ground-truth skeleton",
        "5  Skeleton probability + offset",
        "6  Junction probability",
    )
    panel_size = 390
    header = 70
    footer = 70
    margin = 22
    canvas = Image.new(
        "RGB", (panel_size * len(titles), header + panel_size + footer), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 9), f"Stage 1 encoder | {sample.plant_id} | "
        f"PCL: {sample.metadata.get('pcl_type', 'full')} | front X-Z",
        fill=(20, 20, 20),
    )
    draw.text(
        (12, 31),
        f"semantic mIoU {metrics['semantic_miou']:.3f} | "
        f"skeleton F1 {metrics['skeleton_f1']:.3f} | "
        f"offset MAE {1000 * metrics['centreline_offset_mae_m']:.2f} mm | "
        f"junction F1 {metrics['junction_f1']:.3f}",
        fill=(55, 55, 55),
    )
    draw.text(
        (12, 50),
        f"Probability threshold {probability_threshold:.2f}; cyan points in panel 5 are "
        "high-probability points shifted by the predicted centreline offset.",
        fill=(70, 70, 70),
    )

    valid_indices = sample.point_valid.nonzero(as_tuple=False).flatten()
    render_indices = valid_indices[_sample_indices(len(valid_indices), max_render_points)]
    point_xyz = sample.xyz[render_indices].cpu()
    point_rgb = (sample.rgb[render_indices].cpu().clamp(0, 1) * 255).byte()
    semantic_gt = _semantic_colours(sample.semantic[render_indices])
    predicted_labels = output.semantic_logits[0].argmax(dim=-1).detach().cpu()
    semantic_prediction = _semantic_colours(predicted_labels[render_indices])
    skeleton_probability = output.skeleton_logits[0, :, 0].sigmoid().detach().cpu()
    junction_probability = output.junction_logits[0, :, 0].sigmoid().detach().cpu()
    centreline_offset = output.centreline_offset[0].detach().cpu()
    skeleton_colours = _probability_colours(skeleton_probability[render_indices], "skeleton")
    junction_colours = _probability_colours(junction_probability[render_indices], "junction")

    node_indices = sample.node_valid.nonzero(as_tuple=False).flatten()
    node_xyz = sample.node_xyz[node_indices].cpu()
    bounds_xyz = torch.cat([point_xyz, node_xyz], dim=0) if len(node_xyz) else point_xyz
    bounds = bounds_xyz[:, [0, 2]]
    minimum = bounds.amin(dim=0)
    maximum = bounds.amax(dim=0)
    scale = (panel_size - 2 * margin) / float((maximum - minimum).max().clamp_min(1e-6))
    point_panels = {
        0: point_rgb,
        1: semantic_gt,
        2: semantic_prediction,
        3: torch.full_like(point_rgb, 205),
        4: skeleton_colours,
        5: junction_colours,
    }

    for panel_index, title in enumerate(titles):
        left = panel_index * panel_size
        draw.rectangle(
            (left, header, left + panel_size - 1, header + panel_size - 1),
            outline=(205, 205, 205),
        )
        draw.text((left + 9, header + 8), title, fill=(15, 15, 15))
        pixels = _project(
            point_xyz,
            minimum,
            scale,
            left=left,
            top=header,
            panel_size=panel_size,
            margin=margin,
        )
        _draw_points(draw, pixels, point_panels[panel_index])

        if panel_index == 3 and len(node_xyz):
            node_pixels = _project(
                node_xyz,
                minimum,
                scale,
                left=left,
                top=header,
                panel_size=panel_size,
                margin=margin,
            )
            remap = {old: new for new, old in enumerate(node_indices.tolist())}
            for child in node_indices.tolist():
                parent = int(sample.parent_index[child])
                if parent >= 0 and parent in remap:
                    a = node_pixels[remap[parent]]
                    b = node_pixels[remap[child]]
                    draw.line(
                        (float(a[0]), float(a[1]), float(b[0]), float(b[1])),
                        fill=(220, 45, 35),
                        width=2,
                    )
            for node in node_pixels:
                x, y = float(node[0]), float(node[1])
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(120, 0, 0))

        if panel_index == 4:
            high_probability = sample.point_valid.cpu() & (
                skeleton_probability >= probability_threshold
            )
            high_indices = high_probability.nonzero(as_tuple=False).flatten()
            high_indices = high_indices[_sample_indices(len(high_indices), 10_000)]
            if len(high_indices):
                corrected = (
                    sample.xyz[high_indices].cpu()
                    + centreline_offset[high_indices]
                )
                corrected_pixels = _project(
                    corrected,
                    minimum,
                    scale,
                    left=left,
                    top=header,
                    panel_size=panel_size,
                    margin=margin,
                )
                _draw_points(
                    draw,
                    corrected_pixels,
                    torch.tensor([[0, 255, 255]], dtype=torch.uint8).repeat(
                        len(corrected_pixels), 1
                    ),
                )

        if panel_index == 5 and len(node_xyz):
            junction_nodes = node_indices[
                sample.topology_role[node_indices] == int(TopologyRole.JUNCTION)
            ]
            if len(junction_nodes):
                junction_pixels = _project(
                    sample.node_xyz[junction_nodes].cpu(),
                    minimum,
                    scale,
                    left=left,
                    top=header,
                    panel_size=panel_size,
                    margin=margin,
                )
                for node in junction_pixels:
                    x, y = float(node[0]), float(node[1])
                    draw.ellipse(
                        (x - 5, y - 5, x + 5, y + 5), outline=(0, 255, 255), width=2
                    )

    footer_y = header + panel_size + 12
    x = 14
    for semantic_class, name in SEMANTIC_NAMES.items():
        colour = SEMANTIC_COLOURS[semantic_class]
        draw.rectangle((x, footer_y, x + 13, footer_y + 13), fill=colour)
        draw.text((x + 18, footer_y), name, fill=(40, 40, 40))
        x += 105
    draw.text(
        (panel_size * 3 + 18, footer_y),
        "probability: dark = low, yellow = high | GT skeleton = red | "
        "predicted offset centreline / GT junction markers = cyan",
        fill=(45, 45, 45),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _select_samples(
    dataset: ProcessedPlantDataset, count: int, plant_ids: list[str],
    pcl_type: str = "full",
) -> list[PlantSample]:
    pcl_type = normalise_point_cloud_types([pcl_type])[0]
    if plant_ids:
        requested = set(plant_ids)
        selected = [index for index, entry in enumerate(dataset.instances)
                    if entry["plant_id"] in requested]
        missing = requested - {dataset.instances[index]["plant_id"] for index in selected}
        if missing:
            raise ValueError(f"plant IDs not found in selected split: {sorted(missing)}")
    else:
        if count < 0:
            raise ValueError("--count must be zero (all plants) or a positive integer")
        selected = list(range(len(dataset) if count == 0 else min(count, len(dataset))))
    samples = []
    for index in selected:
        sample = dataset[index]
        if pcl_type == "top_down":
            sample = load_top_down_sample(dataset.paths[index], sample)
        elif pcl_type == "side":
            sample = load_side_sample(dataset.paths[index], sample)
        else:
            sample = replace(sample, metadata={**sample.metadata, "pcl_type": "full"})
        samples.append(sample)
    return samples


def _run_view(
    *,
    pcl_type: str,
    samples: list[PlantSample],
    output_dir: Path,
    model: PointEncoder,
    device: torch.device,
    cfg: Any,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    checkpoint_hash: str,
    processed_root: Path,
    split: str,
    dataset_selection: str,
    skeleton_threshold_m: float,
    junction_threshold_multiplier: float,
    overall_accumulator: EncoderMetricAccumulator,
    event_offset: int,
    event_total: int,
) -> dict[str, Any]:
    """Render and score one selected view into its resolved output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_plant: dict[str, dict[str, float]] = {}
    accumulator = EncoderMetricAccumulator(
        int(cfg.model.encoder.num_semantic_classes),
        skeleton_threshold_m=skeleton_threshold_m,
        junction_threshold_multiplier=junction_threshold_multiplier,
        probability_threshold=args.probability_threshold,
    )
    manifest_path = output_dir / "visualization_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "split": split,
        "dataset": dataset_selection,
        "pcl_types": [pcl_type],
        "expected_plant_ids": [sample.plant_id for sample in samples],
        "renders": {},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            batch = collate_plant_samples([sample]).to(device)
            output = model(
                batch.xyz,
                torch.cat([batch.rgb, batch.normals], dim=-1),
                batch.point_valid,
            )
            metrics = encoder_metrics(
                sample,
                output,
                skeleton_threshold_m=skeleton_threshold_m,
                junction_threshold_multiplier=junction_threshold_multiplier,
                probability_threshold=args.probability_threshold,
            )
            losses = encoder_losses(
                output,
                batch.semantic,
                batch.point_valid,
                batch.node_xyz,
                batch.node_valid,
                batch.topology_role,
                skeleton_threshold_m=skeleton_threshold_m,
                junction_threshold_multiplier=junction_threshold_multiplier,
            )
            accumulator.update(batch, output, losses)
            overall_accumulator.update(batch, output, losses)
            per_plant[sample.plant_id] = metrics
            output_path = output_dir / f"{_safe_filename(sample.plant_id)}.png"
            render_encoder_prediction(
                sample,
                output,
                metrics,
                output_path,
                probability_threshold=args.probability_threshold,
                max_render_points=args.max_render_points,
            )
            render = {
                "path": str(output_path),
                "status": "complete",
                "point_count": int(sample.point_valid.sum()),
            }
            if pcl_type in {"top_down", "side"}:
                render[pcl_type] = sample.metadata[pcl_type]
            manifest["renders"][sample.plant_id] = render
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            current = event_offset + index
            print(f"[{current}/{event_total}] wrote {output_path}", flush=True)
            if os.environ.get("STAGE1_BENCHMARK_EVENTS") == "1":
                print(
                    "@@STAGE1_BENCHMARK_EVENT@@"
                    + json.dumps(
                        {
                            "phase": "visualize",
                            "plant": current,
                            "plant_total": event_total,
                            "plant_id": sample.plant_id,
                            "pcl_type": pcl_type,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "processed_root": str(processed_root),
        "split": split,
        "dataset": dataset_selection,
        "sample_count": len(samples),
        "pcl_types": [pcl_type],
        "view_sample_counts": {pcl_type: len(samples)},
        "probability_threshold": args.probability_threshold,
        "skeleton_threshold_m": skeleton_threshold_m,
        "junction_threshold_multiplier": junction_threshold_multiplier,
        "aggregate": accumulator.compute(),
        "per_plant": per_plant,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/stage1_benchmark/combined/kpconvx/best.ckpt"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        help="Processed cache root; defaults to data.processed_root stored in the checkpoint",
    )
    parser.add_argument("--split", default="test", help="Held-out split (default: test)")
    parser.add_argument(
        "--dataset",
        help="Source dataset ID, or 'combined'; defaults to the checkpoint selection",
    )
    parser.add_argument("--count", type=int, default=3, help="Number of plants; use 0 for all")
    parser.add_argument(
        "--pcl-types",
        nargs="+",
        choices=POINT_CLOUD_TYPES,
        default=["full"],
        help="Ordered point-cloud views used for inference, metrics, and rendering",
    )
    parser.add_argument(
        "--plant-id", action="append", default=[], help="Render this plant ID (repeatable)"
    )
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument(
        "--skeleton-threshold-m",
        type=float,
        help="Override the value stored in the checkpoint configuration",
    )
    parser.add_argument(
        "--junction-threshold-multiplier",
        type=float,
        help="Override the value stored in the checkpoint configuration",
    )
    parser.add_argument("--max-render-points", type=int, default=50_000)
    args = parser.parse_args()
    pcl_types = normalise_point_cloud_types(args.pcl_types)
    if args.output is None:
        suffix = f"_{pcl_types[0]}" if len(pcl_types) == 1 and pcl_types[0] != "full" else ""
        args.output = Path(
            f"outputs/stage1_benchmark/combined/kpconvx/test_visualizations{suffix}"
        )

    if not 0 <= args.probability_threshold <= 1:
        raise ValueError("--probability-threshold must lie in [0, 1]")
    if args.skeleton_threshold_m is not None and args.skeleton_threshold_m <= 0:
        raise ValueError("--skeleton-threshold-m must be positive")
    if (
        args.junction_threshold_multiplier is not None
        and args.junction_threshold_multiplier <= 0
    ):
        raise ValueError("--junction-threshold-multiplier must be positive")

    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if raw_checkpoint.get("stage") != "encoder":
        raise ValueError(
            f"expected an encoder checkpoint, found stage {raw_checkpoint.get('stage')!r}"
        )
    cfg = OmegaConf.create(raw_checkpoint["config"])
    skeleton_threshold_m = float(
        args.skeleton_threshold_m
        if args.skeleton_threshold_m is not None
        else cfg.model.encoder.skeleton_threshold_m
    )
    junction_threshold_multiplier = float(
        args.junction_threshold_multiplier
        if args.junction_threshold_multiplier is not None
        else cfg.model.encoder.junction_threshold_multiplier
    )
    processed_root = args.processed_root or Path(str(cfg.data.processed_root))
    split = str(args.split)
    dataset_selection = normalise_dataset_selection(
        args.dataset or cfg.data.get("dataset", "combined")
    )
    dataset = ProcessedPlantDataset(
        processed_root, split=split, dataset=dataset_selection
    )
    if not len(dataset):
        raise ValueError(f"processed split {split!r} contains no samples at {processed_root}")
    samples_by_view = {
        view: _select_samples(dataset, args.count, args.plant_id, view) for view in pcl_types
    }

    first = samples_by_view[pcl_types[0]][0]
    backbone = create_backbone_from_config(cfg.model.encoder)
    model = PointEncoder(backbone, int(cfg.model.encoder.num_semantic_classes))
    checkpoint = load_checkpoint(
        args.checkpoint,
        model,
        expected_dataset_compatibility=processed_dataset_compatibility(
            processed_root, dataset_selection
        ),
        allow_dataset_subset=True,
        expected_max_nodes=len(first.node_xyz),
    )
    device = _device(args.device)
    model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    total = sum(len(samples) for samples in samples_by_view.values())
    overall_accumulator = EncoderMetricAccumulator(
        int(cfg.model.encoder.num_semantic_classes),
        skeleton_threshold_m=skeleton_threshold_m,
        junction_threshold_multiplier=junction_threshold_multiplier,
        probability_threshold=args.probability_threshold,
    )
    view_reports: dict[str, dict[str, Any]] = {}
    event_offset = 0
    for view in pcl_types:
        view_output = args.output if len(pcl_types) == 1 else args.output / view
        view_reports[view] = _run_view(
            pcl_type=view,
            samples=samples_by_view[view],
            output_dir=view_output,
            model=model,
            device=device,
            cfg=cfg,
            args=args,
            checkpoint=checkpoint,
            checkpoint_hash=checkpoint_hash,
            processed_root=processed_root,
            split=split,
            dataset_selection=dataset_selection,
            skeleton_threshold_m=skeleton_threshold_m,
            junction_threshold_multiplier=junction_threshold_multiplier,
            overall_accumulator=overall_accumulator,
            event_offset=event_offset,
            event_total=total,
        )
        event_offset += len(samples_by_view[view])

    if len(pcl_types) > 1:
        root_manifest = {
            "schema_version": "1.0",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "split": split,
            "dataset": dataset_selection,
            "pcl_types": list(pcl_types),
            "views": {
                view: {
                    "output": str(args.output / view),
                    "sample_count": len(samples_by_view[view]),
                    "status": "complete",
                }
                for view in pcl_types
            },
        }
        (args.output / "visualization_manifest.json").write_text(
            json.dumps(root_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        root_metrics = {
            "schema_version": "1.0",
            "checkpoint": str(args.checkpoint),
            "split": split,
            "dataset": dataset_selection,
            "pcl_types": list(pcl_types),
            "sample_count": total,
            "view_sample_counts": {
                view: len(samples_by_view[view]) for view in pcl_types
            },
            "aggregate": overall_accumulator.compute(),
            "views": view_reports,
        }
        (args.output / "metrics.json").write_text(
            json.dumps(root_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(
        f"rendered {total} {split} view sample(s) on {device}; metrics: "
        f"{args.output / 'metrics.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
