from __future__ import annotations

import sys
import json

import torch
from scipy.optimize import linear_sum_assignment

from tomato_recon.config import load_config
from tomato_recon.data.schemas import SkeletonPrediction
from tomato_recon.evaluation.graph_metrics import graph_metrics
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.models.diffusion.model import ConditionalSkeletonDenoiser
from tomato_recon.models.diffusion.sampling import sample_skeleton
from tomato_recon.models.diffusion.scheduler import DiffusionScheduler
from tomato_recon.models.graph.edge_head import BiologicalGraphModel, graph_training_loss
from tomato_recon.models.graph.decode import decode_plant_graph
from tomato_recon.train.common import (
    create_training_loader,
    epoch_range,
    load_checkpoint,
    load_training_batch,
    maybe_load_upstream,
    seed_everything,
    select_device,
    stage_output_dir,
    save_checkpoint,
    write_metrics,
    write_run_metadata,
)


def align_predicted_nodes(prediction: SkeletonPrediction, batch) -> tuple[torch.Tensor, torch.Tensor]:
    aligned_xyz = batch.node_xyz.clone()
    aligned_flow = batch.parent_flow.clone()
    for batch_index in range(len(batch.plant_ids)):
        predicted_slots = prediction.valid_mask[batch_index].nonzero(as_tuple=False).flatten()
        target_slots = batch.node_valid[batch_index].nonzero(as_tuple=False).flatten()
        if not len(predicted_slots) or not len(target_slots):
            continue
        cost = torch.cdist(
            prediction.node_xyz[batch_index, predicted_slots],
            batch.node_xyz[batch_index, target_slots],
        )
        predicted_row, target_column = linear_sum_assignment(cost.detach().cpu().numpy())
        source = predicted_slots[torch.as_tensor(predicted_row, device=predicted_slots.device)]
        target = target_slots[torch.as_tensor(target_column, device=target_slots.device)]
        aligned_xyz[batch_index, target] = prediction.node_xyz[batch_index, source]
        aligned_flow[batch_index, target] = prediction.parent_flow[batch_index, source]
    return aligned_xyz, aligned_flow


def main(argv: list[str] | None = None) -> None:
    cfg, known = load_config("graph", argv)
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    batch = load_training_batch(cfg, device)
    backbone = create_backbone_from_config(cfg.model.encoder)
    encoder = PointEncoder(backbone, int(cfg.model.encoder.num_semantic_classes)).to(device)
    encoder_hash = maybe_load_upstream(cfg.model.encoder.checkpoint, encoder, batch.samples[0], cfg)
    encoder.requires_grad_(False).eval()
    diffusion = ConditionalSkeletonDenoiser(
        int(cfg.model.diffusion.max_nodes),
        int(cfg.model.encoder.output_dim),
        int(cfg.model.encoder.global_dim),
        int(cfg.model.diffusion.hidden_dim),
        int(cfg.model.diffusion.layers),
        int(cfg.model.diffusion.heads),
        int(cfg.model.diffusion.local_neighbors),
    ).to(device)
    diffusion_hash = maybe_load_upstream(
        cfg.model.diffusion.checkpoint, diffusion, batch.samples[0], cfg
    )
    diffusion.requires_grad_(False).eval()
    scheduler = DiffusionScheduler(int(cfg.model.diffusion.train_timesteps)).to(device)
    model = BiologicalGraphModel(
        int(cfg.model.encoder.output_dim), int(cfg.model.graph.hidden_dim),
        int(cfg.model.graph.knn_candidates), float(cfg.model.graph.max_edge_length_m)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.trainer.learning_rate))
    start_epoch = 0
    if known.resume:
        checkpoint = load_checkpoint(
            known.resume, model, optimizer=optimizer,
            expected_preprocessing_hash=batch.samples[0].metadata.get("preprocessing_hash"),
            expected_max_nodes=int(cfg.data.max_nodes)
        )
        start_epoch = int(checkpoint["epoch"]) + 1
    output_dir = stage_output_dir(cfg, "graph")
    write_run_metadata(cfg, output_dir)
    loader = create_training_loader(cfg)
    best_loss = float("inf")
    metrics = {}
    scores = skeleton = None
    for epoch in epoch_range(cfg, start_epoch):
        totals: dict[str, float] = {}
        steps = 0
        progress = (epoch + 1) / max(int(cfg.trainer.max_epochs), 1)
        noise_std = 0.0 if progress <= 1 / 3 else (0.002 if progress <= 2 / 3 else 0.004)
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            with torch.no_grad():
                encoded = encoder(
                    batch.xyz, torch.cat([batch.rgb, batch.normals], -1), batch.point_valid
                )
            if progress > 2 / 3 and diffusion_hash:
                with torch.no_grad():
                    predicted = sample_skeleton(
                        diffusion,
                        scheduler,
                        encoded,
                        batch.point_valid,
                        sample_steps=int(cfg.model.diffusion.sample_steps),
                        existence_threshold=float(cfg.model.diffusion.existence_threshold),
                        nms_distance_m=float(cfg.model.diffusion.nms_distance_m),
                        min_nodes=int(cfg.model.diffusion.min_nodes),
                        seed=int(cfg.seed) + epoch,
                    )
                noisy_xyz, noisy_flow = align_predicted_nodes(predicted, batch)
            else:
                noisy_xyz = batch.node_xyz + noise_std * torch.randn_like(batch.node_xyz) * batch.node_valid[..., None]
                noisy_flow = batch.parent_flow + noise_std * 20 * torch.randn_like(batch.parent_flow)
            noisy_flow = torch.where(
                batch.parent_flow.norm(dim=-1, keepdim=True) > 0,
                torch.nn.functional.normalize(noisy_flow, dim=-1),
                torch.zeros_like(noisy_flow),
            )
            skeleton = SkeletonPrediction(
                node_xyz=noisy_xyz,
                parent_flow=noisy_flow,
                existence_logit=torch.where(batch.node_valid, 8.0, -8.0),
                confidence=batch.node_valid.float(),
                valid_mask=batch.node_valid,
            )
            scores = model(skeleton, encoded, batch.point_valid)
            loss, parts = graph_training_loss(
                scores,
                batch.organ_type,
                batch.topology_role,
                batch.visibility,
                batch.parent_index,
                batch.node_valid,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = {"loss": loss, **parts}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            steps += 1
            if bool(cfg.trainer.fast_dev_run):
                break
        metrics = {name: value / max(steps, 1) for name, value in totals.items()}
        metrics["curriculum_noise_std_m"] = noise_std
        if metrics["loss"] <= best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(
                output_dir / "best.ckpt",
                stage="graph",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                sample=batch.samples[0],
                epoch=epoch,
                metrics=metrics,
                upstream={
                    name: value
                    for name, value in {"encoder": encoder_hash, "diffusion": diffusion_hash}.items()
                    if value
                },
            )
    assert scores is not None and skeleton is not None
    if batch.samples[0].graph_target is not None:
        predicted_graph = decode_plant_graph(
            batch.plant_ids[0], skeleton, scores, source={"dataset": "training"}
        )
        metrics.update(graph_metrics(predicted_graph, batch.samples[0].graph_target))
        (output_dir / "smoke_graph.json").write_text(
            json.dumps(predicted_graph.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    save_checkpoint(
        output_dir / "best.ckpt",
        stage="graph",
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        sample=batch.samples[0],
        epoch=epoch,
        metrics=metrics,
        upstream={
            name: value
            for name, value in {"encoder": encoder_hash, "diffusion": diffusion_hash}.items()
            if value
        },
    )
    write_metrics(output_dir, metrics)
    print(f"graph checkpoint: {output_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main(sys.argv[1:])
