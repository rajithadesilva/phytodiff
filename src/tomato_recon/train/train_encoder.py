from __future__ import annotations

import sys

import torch

from tomato_recon.config import load_config
from tomato_recon.models.encoders.base import PointEncoder, encoder_losses, nearest_skeleton_targets
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
    write_metrics,
    write_run_metadata,
)


def evaluate_encoder(model, loader, device: torch.device, cfg, epoch: int, epoch_total: int):
    """Evaluate every plant in the validation split and aggregate point-level metrics."""
    totals: dict[str, float] = {}
    steps = 0
    num_classes = int(cfg.model.encoder.num_semantic_classes)
    semantic_intersection = torch.zeros(num_classes, dtype=torch.float64)
    semantic_union = torch.zeros(num_classes, dtype=torch.float64)
    skeleton_true_positive = 0
    skeleton_predicted = 0
    skeleton_target_count = 0
    junction_true_positive = 0
    junction_predicted = 0
    junction_target_count = 0
    offset_absolute_error = 0.0
    offset_element_count = 0
    progress = TrainingProgress("encoder/val", epoch + 1, epoch_total, len(loader))
    model.eval()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            output = model(
                batch.xyz,
                torch.cat([batch.rgb, batch.normals], dim=-1),
                batch.point_valid,
            )
            losses = encoder_losses(
                output,
                batch.semantic,
                batch.point_valid,
                batch.node_xyz,
                batch.node_valid,
                batch.topology_role,
            )
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            steps += 1
            progress.update(totals, steps)

            prediction = output.semantic_logits.argmax(dim=-1)
            labelled = batch.point_valid & (batch.semantic >= 0)
            for semantic_class in range(num_classes):
                predicted_class = prediction == semantic_class
                target_class = batch.semantic == semantic_class
                semantic_intersection[semantic_class] += float(
                    ((predicted_class & target_class) & labelled).sum()
                )
                semantic_union[semantic_class] += float(
                    ((predicted_class | target_class) & labelled).sum()
                )

            skeleton_target, offset_target = nearest_skeleton_targets(
                batch.xyz, batch.node_xyz, batch.node_valid, 0.006
            )
            skeleton_prediction = output.skeleton_logits.squeeze(-1).sigmoid() >= 0.5
            skeleton_true_positive += int(
                (skeleton_prediction & skeleton_target & batch.point_valid).sum()
            )
            skeleton_predicted += int((skeleton_prediction & batch.point_valid).sum())
            skeleton_target_count += int((skeleton_target & batch.point_valid).sum())
            offset_mask = skeleton_target & batch.point_valid
            if offset_mask.any():
                offset_absolute_error += float(
                    (output.centreline_offset[offset_mask] - offset_target[offset_mask])
                    .abs()
                    .sum()
                )
                offset_element_count += int(offset_mask.sum()) * 3

            junction_nodes = batch.node_valid & (batch.topology_role == 2)
            if junction_nodes.any():
                junction_distance = torch.cdist(batch.xyz, batch.node_xyz).masked_fill(
                    ~junction_nodes[:, None], torch.inf
                )
                junction_target = junction_distance.min(dim=-1).values <= 0.009
            else:
                junction_target = torch.zeros_like(batch.point_valid)
            junction_prediction = output.junction_logits.squeeze(-1).sigmoid() >= 0.5
            junction_true_positive += int(
                (junction_prediction & junction_target & batch.point_valid).sum()
            )
            junction_predicted += int((junction_prediction & batch.point_valid).sum())
            junction_target_count += int((junction_target & batch.point_valid).sum())
            if bool(cfg.trainer.fast_dev_run):
                break

    averaged = {name: value / max(steps, 1) for name, value in totals.items()}
    progress.close(averaged)
    ious = [
        float(semantic_intersection[index] / semantic_union[index])
        for index in range(num_classes)
        if semantic_union[index] > 0
    ]
    skeleton_precision = skeleton_true_positive / max(skeleton_predicted, 1)
    skeleton_recall = skeleton_true_positive / max(skeleton_target_count, 1)
    junction_precision = junction_true_positive / max(junction_predicted, 1)
    junction_recall = junction_true_positive / max(junction_target_count, 1)
    averaged.update(
        {
            "semantic_miou": sum(ious) / max(len(ious), 1),
            "skeleton_precision": skeleton_precision,
            "skeleton_recall": skeleton_recall,
            "centreline_offset_mae_m": offset_absolute_error
            / max(offset_element_count, 1),
            "junction_f1": 2
            * junction_precision
            * junction_recall
            / max(junction_precision + junction_recall, 1e-8),
        }
    )
    return averaged


def main(argv: list[str] | None = None) -> None:
    cfg, known = load_config("encoder", argv)
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    batch = load_training_batch(cfg, device)
    backbone = create_backbone_from_config(cfg.model.encoder)
    model = PointEncoder(backbone, int(cfg.model.encoder.num_semantic_classes)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate),
        weight_decay=float(cfg.trainer.weight_decay),
    )
    start_epoch = 0
    best_loss = float("inf")
    if known.resume:
        checkpoint = load_checkpoint(
            known.resume,
            model,
            optimizer=optimizer,
            expected_preprocessing_hash=batch.samples[0].metadata.get("preprocessing_hash"),
            expected_max_nodes=int(cfg.data.max_nodes),
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        resumed_metrics = checkpoint.get("metrics", {})
        best_loss = float(
            resumed_metrics.get(
                "best_val_loss", resumed_metrics.get("val_loss", float("inf"))
            )
        )
    output_dir = stage_output_dir(cfg, "encoder")
    write_run_metadata(cfg, output_dir)
    train_loader = create_training_loader(cfg)
    validation_loader = create_validation_loader(cfg)
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
        validation_metrics = evaluate_encoder(
            model, validation_loader, device, cfg, epoch, epoch_total
        )
        is_best = validation_metrics["loss"] <= best_loss
        best_loss = min(best_loss, validation_metrics["loss"])
        metrics = {
            "loss": validation_metrics["loss"],
            "best_val_loss": best_loss,
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
    best_checkpoint = load_checkpoint(
        output_dir / "best.ckpt",
        model,
        expected_preprocessing_hash=batch.samples[0].metadata.get("preprocessing_hash"),
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
    write_metrics(output_dir, metrics)
    print(
        f"encoder checkpoint: {output_dir / 'best.ckpt'} "
        f"(best val_loss={best_loss:.6f})"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
