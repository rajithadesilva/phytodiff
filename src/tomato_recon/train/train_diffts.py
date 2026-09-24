"""Train one DiffTS stage 2 conditioning variant on all three processed views."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tomato_recon.config import load_config, save_resolved_config
from tomato_recon.data.diffts import augment_z, from_sample, sample_points
from tomato_recon.data.processed import PointCloudViewDataset, processed_dataset_compatibility
from tomato_recon.models.diffusion.diffts_runtime import (
    TRAIN_VIEWS, create_model, load_frozen_stage1, load_stage2_checkpoint,
    save_stage2_checkpoint, stage1_features,
)
from tomato_recon.models.diffusion.diffts_sampling import (
    create_scheduler, node_chamfer, q_sample, sample, skeleton_chamfer,
)
from tomato_recon.train.common import seed_everything, select_device


def _single(items):
    return items[0]


def _loader(cfg, split: str, *, shuffle: bool):
    dataset = PointCloudViewDataset(
        cfg.data.processed_root, split=split, dataset="combined", pcl_types=TRAIN_VIEWS
    )
    return DataLoader(dataset, batch_size=1, shuffle=shuffle,
                      num_workers=int(cfg.trainer.num_workers), collate_fn=_single)


def _condition(model, encoder, data):
    with torch.cuda.amp.autocast(enabled=False):
        point_features, global_features = stage1_features(encoder, data)
    if encoder is not None and data.xyz.is_cuda:
        torch.cuda.empty_cache()
    return model.encode_condition(
        data.xyz, data.rgb, data.rgb_available, point_features, global_features
    )


@torch.no_grad()
def validate(model, encoder, cfg, loader, *, max_samples: int | None = None) -> dict:
    model.eval()
    solver = create_scheduler(
        int(cfg.model.diffusion.train_timesteps),
        float(cfg.model.diffusion.beta_start),
        float(cfg.model.diffusion.beta_end),
    )
    totals = defaultdict(list)
    by_source = defaultdict(lambda: defaultdict(list))
    for index, plant in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        if index % len(TRAIN_VIEWS) != 1:
            continue  # Top-down drives checkpoint selection.
        data = sample_points(from_sample(plant), int(cfg.model.diffusion.max_points),
                             seed=int(cfg.seed) + index).to(next(model.parameters()).device)
        condition = _condition(model, encoder, data)
        prediction = sample(
            model, condition, data.xyz,
            steps=int(cfg.model.diffusion.sample_steps),
            guidance=float(cfg.model.diffusion.guidance),
            seed=int(cfg.seed), scheduler=solver,
        )
        node = float(node_chamfer(prediction.node_xyz, data.target_xyz))
        skeleton = float(skeleton_chamfer(prediction, data.target_xyz, data.target_parent))
        count_error = abs(prediction.predicted_count - len(data.target_xyz))
        for name, value in (("node_chamfer_m", node), ("skeleton_chamfer_m", skeleton),
                            ("count_absolute_error", count_error)):
            totals[name].append(value)
            by_source[data.dataset][name].append(value)
    if not totals:
        raise ValueError("validation has no top-down samples")
    return {
        **{key: sum(values) / len(values) for key, values in totals.items()},
        "top_down_samples": len(totals["node_chamfer_m"]),
        "by_source": {
            source: {key: sum(values) / len(values) for key, values in metrics.items()}
            for source, metrics in by_source.items()
        },
    }


def main(argv: list[str] | None = None) -> None:
    cfg, known = load_config("diffusion", argv)
    if tuple(cfg.data.pcl_types) != TRAIN_VIEWS or str(cfg.data.dataset) != "combined":
        raise ValueError("DiffTS stage 2 training requires combined [full,top_down,side]")
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    if device.type != "cuda":
        raise RuntimeError("DiffTS stage 2 training requires a CUDA GPU")
    train_loader = _loader(cfg, "train", shuffle=True)
    val_loader = _loader(cfg, "val", shuffle=False)
    if len(train_loader.dataset.full_dataset) != 92 or len(val_loader.dataset.full_dataset) != 29:
        raise ValueError("expected all 92 training and 29 validation plants from the three sources")
    compatibility = processed_dataset_compatibility(cfg.data.processed_root, "combined")
    encoder, stage1_hash = load_frozen_stage1(cfg, device)
    model = create_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.trainer.learning_rate),
                                  weight_decay=float(cfg.trainer.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.trainer.mixed_precision))
    scheduler = create_scheduler(
        int(cfg.model.diffusion.train_timesteps),
        float(cfg.model.diffusion.beta_start),
        float(cfg.model.diffusion.beta_end),
    )
    output = Path(str(cfg.output.dir))
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output)
    start_epoch, best, stale = 0, float("inf"), 0
    if known.resume:
        checkpoint = load_stage2_checkpoint(
            known.resume, model, stage1_hash=stage1_hash,
            compatibility=compatibility, optimizer=optimizer,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("metrics", {}).get("best_val_skeleton_chamfer_m", "inf"))
    max_epochs = 1 if bool(cfg.trainer.fast_dev_run) else int(cfg.trainer.max_epochs)
    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        count = 0
        for index, plant in enumerate(train_loader):
            data = from_sample(plant)
            data = sample_points(data, int(cfg.model.diffusion.max_points),
                                 seed=int(cfg.seed) + epoch * 100000 + index)
            generator = torch.Generator().manual_seed(int(cfg.seed) + epoch * 100000 + index)
            data = augment_z(data, generator=generator).to(device)
            with torch.cuda.amp.autocast(enabled=bool(cfg.trainer.mixed_precision)):
                condition = _condition(model, encoder, data)
                timestep = torch.randint(0, int(cfg.model.diffusion.train_timesteps),
                                         (1,), device=device)
                clean = torch.cat((data.target_xyz, data.target_flow), -1)
                noise = torch.randn_like(clean)
                noisy = q_sample(clean, timestep, noise, scheduler)
                conditioned = condition
                if torch.rand(()) < float(cfg.model.diffusion.uncond_probability):
                    from dataclasses import replace
                    conditioned = replace(condition, sparse=condition.sparse.replace_feature(
                        torch.zeros_like(condition.sparse.features)))
                predicted = model.denoise(noisy, timestep, conditioned)
                mse = torch.nn.functional.mse_loss(predicted.float(), noise.float())
                regularization = predicted.float().mean().square() + (predicted.float().std() - 1).square()
                count_loss = torch.nn.functional.cross_entropy(
                    condition.count_logits.float()[None],
                    torch.tensor([len(data.target_xyz) - 2], device=device),
                )
                loss = mse + float(cfg.model.diffusion.regularization_weight) * regularization
                loss = loss + float(cfg.model.diffusion.count_loss_weight) * count_loss
            scaler.scale(loss / int(cfg.trainer.gradient_accumulation)).backward()
            if (index + 1) % int(cfg.trainer.gradient_accumulation) == 0 or index + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for name, value in (("loss", loss), ("mse", mse), ("count", count_loss)):
                totals[name] += float(value.detach())
            count += 1
            if bool(cfg.trainer.fast_dev_run):
                break
        summary = {name: value / count for name, value in totals.items()}
        print(f"DiffTS {model.variant} epoch {epoch + 1}: {summary}", flush=True)
        validate_now = bool(cfg.trainer.fast_dev_run) or (epoch + 1) % int(cfg.trainer.validation_interval) == 0
        if validate_now:
            validation = validate(model, encoder, cfg, val_loader,
                                  max_samples=2 if bool(cfg.trainer.fast_dev_run) else None)
            summary.update({f"val_{key}": value for key, value in validation.items()
                            if isinstance(value, (int, float))})
            score = validation["skeleton_chamfer_m"]
            if score < best:
                best, stale = score, 0
                summary["best_val_skeleton_chamfer_m"] = best
                save_stage2_checkpoint(output / "best.ckpt", model, optimizer, cfg,
                                       epoch, summary, stage1_hash, compatibility)
            else:
                stale += 1
            (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        summary["best_val_skeleton_chamfer_m"] = best
        save_stage2_checkpoint(output / "last.ckpt", model, optimizer, cfg,
                               epoch, summary, stage1_hash, compatibility)
        (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        if stale >= int(cfg.trainer.early_stop_patience):
            print(f"DiffTS {model.variant}: validation patience reached", flush=True)
            break
    print(f"DiffTS checkpoint: {output / 'best.ckpt'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
