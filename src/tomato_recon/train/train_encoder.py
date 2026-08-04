from __future__ import annotations

import sys

import torch

from tomato_recon.config import load_config
from tomato_recon.models.encoders.base import PointEncoder, encoder_losses, nearest_skeleton_targets
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.train.common import (
    TrainingProgress,
    create_training_loader,
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


def main(argv: list[str] | None = None) -> None:
    cfg, known = load_config("encoder", argv)
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    batch = load_training_batch(cfg, device)
    backbone = create_backbone_from_config(cfg.model.encoder)
    model = PointEncoder(backbone, int(cfg.model.encoder.num_semantic_classes)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.trainer.learning_rate), weight_decay=float(cfg.trainer.weight_decay)
    )
    start_epoch = 0
    if known.resume:
        checkpoint = load_checkpoint(
            known.resume,
            model,
            optimizer=optimizer,
            expected_preprocessing_hash=batch.samples[0].metadata.get("preprocessing_hash"),
            expected_max_nodes=int(cfg.data.max_nodes),
        )
        start_epoch = int(checkpoint["epoch"]) + 1
    output_dir = stage_output_dir(cfg, "encoder")
    write_run_metadata(cfg, output_dir)
    loader = create_training_loader(cfg)
    best_loss = float("inf")
    metrics = {}
    output = None
    epoch_total = (
        start_epoch + 1
        if bool(cfg.trainer.fast_dev_run)
        else max(int(cfg.trainer.max_epochs), start_epoch + 1)
    )
    for epoch in epoch_range(cfg, start_epoch):
        totals: dict[str, float] = {}
        steps = 0
        progress = TrainingProgress("encoder", epoch + 1, epoch_total, len(loader))
        for cpu_batch in loader:
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
        metrics = {name: value / max(steps, 1) for name, value in totals.items()}
        progress.close(metrics)
        if metrics["loss"] <= best_loss:
            best_loss = metrics["loss"]
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
    assert output is not None
    with torch.no_grad():
        prediction = output.semantic_logits.argmax(-1)
        valid = batch.point_valid & (batch.semantic >= 0)
        ious = []
        for semantic_class in range(int(cfg.model.encoder.num_semantic_classes)):
            predicted_class = prediction == semantic_class
            target_class = batch.semantic == semantic_class
            union = ((predicted_class | target_class) & valid).sum()
            if union:
                ious.append(float(((predicted_class & target_class) & valid).sum() / union))
        skeleton_target, offset_target = nearest_skeleton_targets(
            batch.xyz, batch.node_xyz, batch.node_valid, 0.006
        )
        skeleton_prediction = output.skeleton_logits.squeeze(-1).sigmoid() >= 0.5
        true_positive = (skeleton_prediction & skeleton_target & batch.point_valid).sum()
        offset_mask = skeleton_target & batch.point_valid
        junction_nodes = batch.node_valid & (batch.topology_role == 2)
        junction_distance = torch.cdist(batch.xyz, batch.node_xyz).masked_fill(
            ~junction_nodes[:, None], torch.inf
        )
        junction_target = junction_distance.min(-1).values <= 0.009
        junction_prediction = output.junction_logits.squeeze(-1).sigmoid() >= 0.5
        junction_tp = (junction_prediction & junction_target & batch.point_valid).sum()
        junction_precision = junction_tp / (junction_prediction & batch.point_valid).sum().clamp_min(1)
        junction_recall = junction_tp / (junction_target & batch.point_valid).sum().clamp_min(1)
        metrics.update(
            {
                "semantic_miou": sum(ious) / max(len(ious), 1),
                "skeleton_precision": float(
                    true_positive / (skeleton_prediction & batch.point_valid).sum().clamp_min(1)
                ),
                "skeleton_recall": float(
                    true_positive / (skeleton_target & batch.point_valid).sum().clamp_min(1)
                ),
                "centreline_offset_mae_m": float(
                    (output.centreline_offset[offset_mask] - offset_target[offset_mask]).abs().mean()
                    if offset_mask.any()
                    else 0.0
                ),
                "junction_f1": float(
                    2 * junction_precision * junction_recall
                    / (junction_precision + junction_recall).clamp_min(1e-8)
                ),
            }
        )
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
    torch.save(
        {
            "schema_version": "1.0",
            "plant_ids": batch.plant_ids,
            "point_features": output.point_features.detach().cpu(),
            "semantic_logits": output.semantic_logits.detach().cpu(),
            "skeleton_logits": output.skeleton_logits.detach().cpu(),
        },
        output_dir / "smoke_predictions.pt",
    )
    write_metrics(output_dir, metrics)
    print(f"encoder checkpoint: {output_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main(sys.argv[1:])
