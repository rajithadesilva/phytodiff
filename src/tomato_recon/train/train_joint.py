from __future__ import annotations

import json
import sys

import torch

from tomato_recon.config import load_config
from tomato_recon.data.schemas import IGNORE_INDEX, SemanticClass, SkeletonPrediction
from tomato_recon.models.parametric.decoder import parametric_training_loss
from tomato_recon.models.parametric.losses import (
    normal_consistency_loss,
    occlusion_aware_geometry_loss,
    positive_radius_regularisation,
)
from tomato_recon.models.parametric.primitives import generate_plant_geometry
from tomato_recon.models.pipeline import TomatoReconstructionPipeline
from tomato_recon.train.train_parametric import fitted_parameter_tensor
from tomato_recon.train.common import (
    TrainingProgress,
    create_training_loader,
    epoch_range,
    load_checkpoint,
    load_training_batch,
    maybe_load_upstream,
    selected_point_cloud_types,
    seed_everything,
    select_device,
    stage_output_dir,
    save_checkpoint,
    training_dataset_compatibility,
    write_metrics,
    write_run_metadata,
)


def _limit_rows(*values: torch.Tensor, limit: int) -> tuple[torch.Tensor, ...]:
    if len(values[0]) <= limit:
        return values
    index = torch.linspace(0, len(values[0]) - 1, limit, device=values[0].device).long()
    return tuple(value[index] for value in values)


def main(argv: list[str] | None = None) -> None:
    cfg, known = load_config("joint", argv)
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    batch = load_training_batch(cfg, device)
    model = TomatoReconstructionPipeline(cfg).to(device)
    upstream = {}
    for name, module in (
        ("encoder", model.encoder), ("diffusion", model.denoiser),
        ("graph", model.graph_model), ("parametric", model.parametric_decoder)
    ):
        path = cfg.model[name].get("checkpoint")
        digest = maybe_load_upstream(path, module, batch.samples[0], cfg)
        if digest:
            upstream[name] = digest
    model.encoder.requires_grad_(False)
    model.denoiser.requires_grad_(False)
    # Discrete graph decoding remains stop-gradient; tune continuous parameter heads conservatively.
    parameters = list(model.graph_model.node_head.parameters()) + list(model.parametric_decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(cfg.trainer.learning_rate))
    start_epoch = 0
    if known.resume:
        checkpoint = load_checkpoint(
            known.resume, model, optimizer=optimizer,
            expected_dataset_compatibility=training_dataset_compatibility(
                cfg, batch.samples[0]
            ),
            expected_pcl_types=selected_point_cloud_types(cfg),
            expected_max_nodes=int(cfg.data.max_nodes)
        )
        start_epoch = int(checkpoint["epoch"]) + 1
    output_dir = stage_output_dir(cfg, "joint")
    write_run_metadata(cfg, output_dir)
    model.eval()
    staged_results = model.reconstruct(batch)
    staged_validation = {
        "schema_version": "1.0",
        "valid": True,
        "plant_ids": batch.plant_ids,
        "node_counts": [len(result.graph.nodes) for result in staged_results],
        "mesh_counts": [len(result.geometry.meshes) for result in staged_results],
        "upstream_checkpoint_hashes": upstream,
    }
    (output_dir / "staged_validation.json").write_text(
        json.dumps(staged_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model.train()
    model.encoder.eval()
    model.denoiser.eval()
    loader = create_training_loader(cfg)
    best_loss = float("inf")
    metrics = {}
    final_geometry = None
    epoch_total = (
        start_epoch + 1
        if bool(cfg.trainer.fast_dev_run)
        else max(int(cfg.trainer.max_epochs), start_epoch + 1)
    )
    for epoch in epoch_range(cfg, start_epoch):
        totals: dict[str, float] = {}
        steps = 0
        progress = TrainingProgress("joint", epoch + 1, epoch_total, len(loader))
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            with torch.no_grad():
                encoded = model.encoder(
                    batch.xyz, torch.cat([batch.rgb, batch.normals], -1), batch.point_valid
                )
            # Continuous objective only; discrete arborescence decoding remains stop-gradient.
            skeleton = SkeletonPrediction(
                node_xyz=batch.node_xyz,
                parent_flow=batch.parent_flow,
                existence_logit=torch.where(batch.node_valid, 8.0, -8.0),
                confidence=batch.node_valid.float(),
                valid_mask=batch.node_valid,
            )
            graph_scores = model.graph_model(skeleton, encoded, batch.point_valid)
            raw = model.parametric_decoder(graph_scores.nodes.embedding)
            target = fitted_parameter_tensor(batch, device)
            parametric = parametric_training_loss(raw, target, batch.node_valid)
            geometry_total = raw.sum() * 0
            normal_total = raw.sum() * 0
            skeleton_total = raw.sum() * 0
            radius_total = raw.sum() * 0
            for batch_index, sample in enumerate(batch.samples):
                if sample.graph_target is None:
                    raise ValueError("Stage 5 requires graph targets for the stop-gradient decode")
                decoded = model.parametric_decoder.decode_graph(
                    sample.graph_target, raw, batch=batch_index
                )
                geometry = generate_plant_geometry(
                    decoded,
                    curve_samples=int(cfg.model.parametric.curve_samples),
                    radial_segments=int(cfg.model.parametric.radial_segments),
                )
                final_geometry = geometry
                model_points = torch.cat([mesh.vertices for mesh in geometry.meshes])
                model_normals = torch.cat(
                    [
                        mesh.normals
                        if mesh.normals is not None
                        else torch.zeros_like(mesh.vertices)
                        for mesh in geometry.meshes
                    ]
                )
                if geometry.visibility_weights is None:
                    raise RuntimeError("geometry generator did not provide visibility weights")
                model_points, model_normals, visibility_weight = _limit_rows(
                    model_points,
                    model_normals,
                    geometry.visibility_weights,
                    limit=int(cfg.trainer.geometry_points),
                )
                plant_point = (
                    batch.point_valid[batch_index]
                    & (batch.semantic[batch_index] != int(SemanticClass.BACKGROUND))
                    & (batch.semantic[batch_index] != int(SemanticClass.SUPPORT_POLE))
                    & (batch.semantic[batch_index] != IGNORE_INDEX)
                )
                if not plant_point.any():
                    plant_point = batch.point_valid[batch_index]
                scan_points, scan_normals = _limit_rows(
                    batch.xyz[batch_index][plant_point],
                    batch.normals[batch_index][plant_point],
                    limit=int(cfg.trainer.geometry_points),
                )
                geometry_total = geometry_total + occlusion_aware_geometry_loss(
                    scan_points, model_points, visibility_weight
                )
                normal_total = normal_total + normal_consistency_loss(
                    scan_points,
                    scan_normals,
                    model_points,
                    model_normals,
                    visibility_weight,
                )
                if geometry.skeleton_vertices is not None:
                    skeleton_distance = torch.cdist(
                        geometry.skeleton_vertices,
                        batch.node_xyz[batch_index][batch.node_valid[batch_index]],
                    ).square()
                    skeleton_total = skeleton_total + skeleton_distance.min(dim=1).values.mean()
                radii = [
                    torch.as_tensor(radius, device=device)
                    for organ in decoded.organs
                    for radius in (organ.radius_start_m, organ.radius_end_m)
                    if radius is not None
                ]
                if radii:
                    radius_total = radius_total + positive_radius_regularisation(
                        torch.stack(radii)
                    )
            batch_count = max(len(batch.samples), 1)
            geometry_total = geometry_total / batch_count
            normal_total = normal_total / batch_count
            skeleton_total = skeleton_total / batch_count
            radius_total = radius_total / batch_count
            loss = (
                float(cfg.loss.parametric_weight) * parametric
                + float(cfg.loss.geometry_weight) * geometry_total
                + float(cfg.loss.normal_weight) * normal_total
                + float(cfg.loss.skeleton_weight) * skeleton_total
                + float(cfg.loss.radius_weight) * radius_total
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = {
                "loss": loss,
                "parametric": parametric,
                "occlusion_aware_geometry": geometry_total,
                "normal_consistency": normal_total,
                "skeleton_consistency": skeleton_total,
                "radius_regularisation": radius_total,
            }
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            steps += 1
            progress.update(totals, steps)
            if bool(cfg.trainer.fast_dev_run):
                break
        metrics = {name: value / max(steps, 1) for name, value in totals.items()}
        metrics["discrete_decode_gradient"] = 0.0
        metrics["staged_pipeline_valid"] = 1.0
        progress.close(metrics)
        if metrics["loss"] <= best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(
                output_dir / "best.ckpt",
                stage="joint",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                sample=batch.samples[0],
                epoch=epoch,
                metrics=metrics,
                upstream=upstream,
            )
    if final_geometry is not None and final_geometry.visibility_weights is not None:
        torch.save(
            {
                "schema_version": "1.0",
                "vertices": final_geometry.combined_mesh().vertices.detach().cpu(),
                "visibility_weight": final_geometry.visibility_weights.detach().cpu(),
            },
            output_dir / "smoke_visibility_weights.pt",
        )
    write_metrics(output_dir, metrics)
    print(f"joint checkpoint: {output_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main(sys.argv[1:])
