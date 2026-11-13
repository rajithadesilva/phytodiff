"""Train AG-GFM on selected source datasets and views, choosing best on validation."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tomato_recon.config import load_config, save_resolved_config
from tomato_recon.data.aggfm import dataset_compatibility, from_sample, sample_points
from tomato_recon.data.processed import PointCloudViewDataset
from tomato_recon.evaluation.aggfm import evaluate_dataset
from tomato_recon.models.flow_matching.losses import training_loss
from tomato_recon.models.flow_matching.runtime import (
    create_model, create_stage1, extract_stage1, load_checkpoint, require_schema, save_checkpoint,
)
from tomato_recon.train.common import TrainingProgress, seed_everything, select_device


def validation_view(cfg):
    if "top_down" in cfg.data.pcl_types:
        return "top_down"
    return str(cfg.trainer.validation_view or cfg.data.pcl_types[0])


def make_dataset(cfg, split, views=None):
    return PointCloudViewDataset(cfg.data.processed_root, split=split, dataset=cfg.data.dataset,
                                 pcl_types=tuple(cfg.data.pcl_types) if views is None else views)


def check_split_isolation(train, val):
    """Also check repeated scans of the same source plant across every manifest split."""
    groups = {}
    for item in train.full_dataset.manifest["instances"]:
        if item.get("status") != "complete":
            continue
        if train.dataset != "combined" and item["dataset"] != train.dataset:
            continue
        key = (item["dataset"], item["source_plant_id"])
        split = item.get("split")
        if key in groups and groups[key] != split:
            raise ValueError(f"plant-level split leakage for {key}")
        groups[key] = split
    if not len(train) or not len(val):
        raise ValueError("selected dataset needs nonempty train and validation splits")


def main(argv=None):
    cfg, known = load_config("aggfm", argv)
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    train = make_dataset(cfg, "train")
    val = make_dataset(cfg, "val", (validation_view(cfg),))
    check_split_isolation(train, val)
    loader = DataLoader(train, batch_size=int(cfg.trainer.batch_size), shuffle=True,
                        num_workers=int(cfg.trainer.num_workers), collate_fn=list)
    compatibility = dataset_compatibility(cfg.data.processed_root, str(cfg.data.dataset))
    resume = torch.load(known.resume, map_location="cpu", weights_only=False) if known.resume else None
    if resume is not None:
        require_schema(resume)
    encoder, stage1_hash = create_stage1(cfg, device, compatibility=compatibility, source=resume is None)
    model = create_model(cfg).to(device)
    frozen = bool(cfg.trainer.freeze_stage1)
    encoder.requires_grad_(not frozen)
    groups = [{"params": model.parameters(), "lr": float(cfg.trainer.learning_rate)},
              {"params": encoder.parameters(), "lr": float(cfg.trainer.stage1_learning_rate)}]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg.trainer.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    output = Path(str(cfg.output.dir))
    if output == Path(f"outputs/aggfm/{cfg.data.dataset}"):
        output = output / "_".join(cfg.data.pcl_types)
        cfg.output.dir = str(output)
    if not known.resume and any((output / name).exists() for name in ("last.ckpt", "best.ckpt")):
        raise FileExistsError(f"{output} contains checkpoints; use --resume or a new output.dir")
    start, best, stale = 0, float("inf"), 0
    if resume is not None:
        load_checkpoint(known.resume, model, encoder, cfg, compatibility=compatibility,
                         optimizer=optimizer, scheduler=scheduler, resume=True, payload=resume)
        if resume["dataset_compatibility"].get("split_signature") != compatibility["split_signature"]:
            raise ValueError("AG-GFM resume split assignment mismatch")
        start, best = int(resume["epoch"]) + 1, float(resume["metrics"]["best_val_skeleton_chamfer_m"])
        stale, stage1_hash = int(resume.get("stale", 0)), resume["stage1_checkpoint_sha256"]
    save_resolved_config(cfg, output)
    epochs = start + 1 if cfg.trainer.fast_dev_run else int(cfg.trainer.max_epochs)
    for epoch in range(start, epochs):
        model.train()
        encoder.train(not frozen)
        totals, count = {}, 0
        batch_total = 1 if cfg.trainer.fast_dev_run else len(loader)
        print(f"[aggfm/train] Epoch {epoch + 1}/{epochs} | Starting training | "
              f"batches={batch_total} | device={device}", flush=True)
        progress = TrainingProgress("aggfm/train", epoch + 1, epochs, batch_total, log_interval=1)
        for batch_index, plants in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            for plant in plants:
                data = sample_points(from_sample(plant), int(cfg.model.aggfm.max_points),
                                     int(cfg.seed) + epoch * len(train) + count).to(device)
                condition = extract_stage1(encoder, data.points_only(), frozen=frozen)
                seeds, memory = model.condition(condition)
                losses, diagnostics = training_loss(model, seeds, memory, data, cfg.model.aggfm)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"non-finite loss for {data.plant_id}")
                (losses["loss"] / len(plants)).backward()
                for key, value in {**losses, **{f"match_{k}": v for k, v in diagnostics.items()}}.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach() if isinstance(value, torch.Tensor) else value)
                count += 1
            torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]],
                                           float(cfg.trainer.gradient_clip), error_if_nonfinite=True)
            optimizer.step()
            progress.update({"loss": totals["loss"]}, batch_index + 1, sample_count=count)
            if cfg.trainer.fast_dev_run:
                break
        summary = {key: value/count for key, value in totals.items()}
        progress.close({"loss": summary["loss"]})
        validate_now = cfg.trainer.fast_dev_run or (epoch + 1) % int(cfg.trainer.validation_interval) == 0
        improved = False
        if validate_now:
            validation_count = 1 if cfg.trainer.fast_dev_run else len(val)
            print(f"[aggfm/val] Epoch {epoch + 1}/{epochs} | Validating {validation_count} samples "
                  f"(view={validation_view(cfg)})", flush=True)
            validation = evaluate_dataset(model, encoder, cfg, val,
                                           max_samples=1 if cfg.trainer.fast_dev_run else None,
                                           progress_label=f"[aggfm/val] Epoch {epoch + 1}/{epochs}")
            score = validation["overall"]["skeleton_chamfer_m"]
            scheduler.step(score)
            improved = score < best
            best, stale = (score, 0) if improved else (best, stale + 1)
            summary.update({f"val_{k}": v for k, v in validation["overall"].items()})
            (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        summary["best_val_skeleton_chamfer_m"] = best
        summary["selection_view"] = validation_view(cfg)
        for name in (["last.ckpt", "best.ckpt"] if improved else ["last.ckpt"]):
            save_checkpoint(output / name, model, encoder, optimizer, cfg, epoch, summary,
                             stage1_hash, compatibility, scheduler=scheduler, stale=stale)
        (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"AG-GFM epoch {epoch + 1}: loss={summary['loss']:.5f}; best validation={best:.6f} m", flush=True)
        if stale >= int(cfg.trainer.early_stop_patience):
            break


if __name__ == "__main__":
    main()
