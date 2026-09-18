from __future__ import annotations

import json
import os
import sys
import time

import torch

from tomato_recon.config import load_config
from tomato_recon.evaluation.encoder import evaluate_encoder_model
from tomato_recon.models.encoders.base import PointEncoder, encoder_losses
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.train.common import (
    TrainingProgress,
    create_training_loader,
    create_validation_loader,
    epoch_range,
    load_checkpoint,
    load_training_batch,
    seed_everything,
    select_device,
    stage_output_dir,
    save_checkpoint,
    selected_point_cloud_types,
    training_dataset_compatibility,
    write_metrics,
    write_run_metadata,
)


def _benchmark_event(**values: object) -> None:
    if os.environ.get("STAGE1_BENCHMARK_EVENTS") == "1":
        print("@@STAGE1_BENCHMARK_EVENT@@" + json.dumps(values, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> None:
    training_started = time.perf_counter()
    cfg, known = load_config("encoder", argv)
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    batch = load_training_batch(cfg, device)
    backbone = create_backbone_from_config(cfg.model.encoder)
    model = PointEncoder(backbone, int(cfg.model.encoder.num_semantic_classes)).to(device)
    skeleton_threshold_m = float(cfg.model.encoder.skeleton_threshold_m)
    junction_threshold_multiplier = float(
        cfg.model.encoder.junction_threshold_multiplier
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )
    start_epoch = 0
    best_score = float("-inf")
    if known.resume:
        checkpoint = load_checkpoint(
            known.resume,
            model,
            optimizer=optimizer,
            expected_dataset_compatibility=training_dataset_compatibility(
                cfg, batch.samples[0]
            ),
            expected_pcl_types=selected_point_cloud_types(cfg),
            expected_max_nodes=int(cfg.data.max_nodes),
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        resumed_metrics = checkpoint.get("metrics", {})
        best_score = float(
            resumed_metrics.get(
                "best_val_overall_score",
                resumed_metrics.get("val_overall_score", float("-inf")),
            )
        )
    output_dir = stage_output_dir(cfg, "encoder")
    write_run_metadata(cfg, output_dir)
    train_loader = create_training_loader(cfg)
    validation_loader = create_validation_loader(cfg)
    pcl_types = selected_point_cloud_types(cfg)
    train_plant_count = (
        len(train_loader.dataset.full_dataset)
        if hasattr(train_loader, "dataset")
        else 1
    )
    validation_plant_count = (
        len(validation_loader.dataset.full_dataset)
        if hasattr(validation_loader, "dataset")
        else 1
    )
    epoch_total = (
        start_epoch + 1
        if bool(cfg.trainer.fast_dev_run)
        else max(int(cfg.trainer.max_epochs), start_epoch + 1)
    )
    for epoch in epoch_range(cfg, start_epoch):
        totals: dict[str, float] = {}
        steps = 0
        progress = TrainingProgress("encoder/train", epoch + 1, epoch_total, len(train_loader))
        for cpu_batch in train_loader:
            batch = cpu_batch.to(device)
            model.train()
            output = model(
                batch.xyz, torch.cat([batch.rgb, batch.normals], dim=-1), batch.point_valid
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
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            steps += 1
            progress.update(totals, steps)
            if bool(cfg.trainer.fast_dev_run):
                break
        train_metrics = {name: value / max(steps, 1) for name, value in totals.items()}
        progress.close(train_metrics)
        validation_metrics = evaluate_encoder_model(
            model,
            validation_loader,
            device,
            num_classes=int(cfg.model.encoder.num_semantic_classes),
            skeleton_threshold_m=skeleton_threshold_m,
            junction_threshold_multiplier=junction_threshold_multiplier,
            epoch=epoch,
            epoch_total=epoch_total,
            fast_dev_run=bool(cfg.trainer.fast_dev_run),
        )
        is_best = validation_metrics["overall_score"] >= best_score
        best_score = max(best_score, validation_metrics["overall_score"])
        metrics = {
            "loss": validation_metrics["loss"],
            "best_val_overall_score": best_score,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in validation_metrics.items()},
        }
        save_checkpoint(
            output_dir / "last.ckpt",
            stage="encoder",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            sample=batch.samples[0],
            epoch=epoch,
            metrics=metrics,
        )
        if is_best:
            save_checkpoint(
                output_dir / "best.ckpt",
                stage="encoder",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                sample=batch.samples[0],
                epoch=epoch,
                metrics=metrics,
            )
        _benchmark_event(
            phase="train",
            epoch=epoch + 1,
            epoch_total=epoch_total,
            validation_overall_score=validation_metrics["overall_score"],
        )
    best_checkpoint = load_checkpoint(
        output_dir / "best.ckpt",
        model,
        expected_dataset_compatibility=training_dataset_compatibility(
            cfg, batch.samples[0]
        ),
        expected_pcl_types=selected_point_cloud_types(cfg),
        expected_max_nodes=int(cfg.data.max_nodes),
    )
    metrics = best_checkpoint["metrics"]
    validation_batch = next(iter(validation_loader)).to(device)
    model.eval()
    with torch.inference_mode():
        output = model(
            validation_batch.xyz,
            torch.cat([validation_batch.rgb, validation_batch.normals], dim=-1),
            validation_batch.point_valid,
        )
    torch.save(
        {
            "schema_version": "1.0",
            "split": str(cfg.trainer.validation_split),
            "plant_ids": validation_batch.plant_ids,
            "point_features": output.point_features.detach().cpu(),
            "semantic_logits": output.semantic_logits.detach().cpu(),
            "skeleton_logits": output.skeleton_logits.detach().cpu(),
            "centreline_offset": output.centreline_offset.detach().cpu(),
            "junction_logits": output.junction_logits.detach().cpu(),
        },
        output_dir / "smoke_predictions.pt",
    )
    metrics = dict(metrics)
    metrics.update(
        {
            "best_epoch": int(best_checkpoint.get("epoch", -1)) + 1,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "training_runtime_seconds": time.perf_counter() - training_started,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if device.type == "cuda"
                else 0.0
            ),
            "pcl_types": list(pcl_types),
            "training_view_sample_counts": {
                view: train_plant_count for view in pcl_types
            },
            "validation_view_sample_counts": {
                view: validation_plant_count for view in pcl_types
            },
        }
    )
    write_metrics(output_dir, metrics)
    print(
        f"encoder checkpoint: {output_dir / 'best.ckpt'} "
        f"(best val overall={best_score:.6f})"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
