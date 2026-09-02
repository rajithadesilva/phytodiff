from __future__ import annotations

import sys

import torch

from tomato_recon.config import load_config
from tomato_recon.models.diffusion.losses import diffusion_training_loss
from tomato_recon.models.diffusion.model import ConditionalSkeletonDenoiser
from tomato_recon.models.diffusion.scheduler import DiffusionScheduler
from tomato_recon.models.diffusion.sampling import sample_skeleton
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.train.common import (
    TrainingProgress,
    create_training_loader,
    epoch_range,
    load_checkpoint,
    load_training_batch,
    maybe_load_upstream,
    seed_everything,
    select_device,
    stage_output_dir,
    save_checkpoint,
    training_dataset_compatibility,
    write_metrics,
    write_run_metadata,
)


def main(argv: list[str] | None = None) -> None:
    cfg, known = load_config("diffusion", argv)
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    batch = load_training_batch(cfg, device)
    backbone = create_backbone_from_config(cfg.model.encoder)
    encoder = PointEncoder(backbone, int(cfg.model.encoder.num_semantic_classes)).to(device)
    upstream_hash = maybe_load_upstream(cfg.model.encoder.checkpoint, encoder, batch.samples[0], cfg)
    if bool(cfg.model.diffusion.freeze_encoder):
        encoder.requires_grad_(False).eval()
    model = ConditionalSkeletonDenoiser(
        int(cfg.model.diffusion.max_nodes),
        int(cfg.model.encoder.output_dim),
        int(cfg.model.encoder.global_dim),
        int(cfg.model.diffusion.hidden_dim),
        int(cfg.model.diffusion.layers),
        int(cfg.model.diffusion.heads),
        int(cfg.model.diffusion.local_neighbors),
    ).to(device)
    scheduler = DiffusionScheduler(int(cfg.model.diffusion.train_timesteps)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.trainer.learning_rate))
    start_epoch = 0
    if known.resume:
        checkpoint = load_checkpoint(
            known.resume,
            model,
            optimizer=optimizer,
            expected_dataset_compatibility=training_dataset_compatibility(
                cfg, batch.samples[0]
            ),
            expected_max_nodes=int(cfg.data.max_nodes),
        )
        start_epoch = int(checkpoint["epoch"]) + 1
    output_dir = stage_output_dir(cfg, "diffusion")
    write_run_metadata(cfg, output_dir)
    loader = create_training_loader(cfg)
    best_loss = float("inf")
    metrics = {}
    epoch_total = (
        start_epoch + 1
        if bool(cfg.trainer.fast_dev_run)
        else max(int(cfg.trainer.max_epochs), start_epoch + 1)
    )
    for epoch in epoch_range(cfg, start_epoch):
        totals: dict[str, float] = {}
        steps = 0
        progress = TrainingProgress("diffusion", epoch + 1, epoch_total, len(loader))
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            with torch.set_grad_enabled(not bool(cfg.model.diffusion.freeze_encoder)):
                encoded = encoder(
                    batch.xyz, torch.cat([batch.rgb, batch.normals], -1), batch.point_valid
                )
            timestep = torch.randint(
                0, scheduler.train_timesteps, (len(batch.plant_ids),), device=device
            )
            loss, parts, _ = diffusion_training_loss(
                model,
                scheduler,
                encoded,
                batch.point_valid,
                batch.node_xyz,
                batch.parent_flow,
                batch.node_valid,
                timestep,
                visibility=batch.visibility,
                duplicate_sigma=float(cfg.model.diffusion.nms_distance_m),
                existence_weight=float(cfg.loss.existence_weight),
                confidence_weight=float(cfg.loss.confidence_weight),
                flow_weight=float(cfg.loss.flow_weight),
                duplicate_weight=float(cfg.loss.duplicate_weight),
                bounds_weight=float(cfg.loss.bounds_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = {"loss": loss, **parts}
            for name, value in values.items():
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
                stage="diffusion",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                sample=batch.samples[0],
                epoch=epoch,
                metrics=metrics,
                upstream={"encoder": upstream_hash} if upstream_hash else {},
            )
    with torch.no_grad():
        prediction = sample_skeleton(
            model,
            scheduler,
            encoded,
            batch.point_valid,
            sample_steps=int(cfg.model.diffusion.sample_steps),
            existence_threshold=float(cfg.model.diffusion.existence_threshold),
            nms_distance_m=float(cfg.model.diffusion.nms_distance_m),
            min_nodes=int(cfg.model.diffusion.min_nodes),
            seed=int(cfg.seed),
        )
    torch.save(
        {
            "schema_version": "1.0",
            "plant_ids": batch.plant_ids,
            "node_xyz": prediction.node_xyz.detach().cpu(),
            "parent_flow": prediction.parent_flow.detach().cpu(),
            "existence_logit": prediction.existence_logit.detach().cpu(),
            "confidence": prediction.confidence.detach().cpu(),
            "valid_mask": prediction.valid_mask.detach().cpu(),
        },
        output_dir / "smoke_predictions.pt",
    )
    write_metrics(output_dir, metrics)
    print(f"diffusion checkpoint: {output_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main(sys.argv[1:])
