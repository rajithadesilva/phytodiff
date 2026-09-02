from __future__ import annotations

import sys
import json

import torch

from tomato_recon.config import load_config
from tomato_recon.data.preprocess import fit_parametric_targets
from tomato_recon.data.schemas import SkeletonPrediction
from tomato_recon.models.parametric.decoder import PARAMETER_WIDTH, ParametricDecoder, parametric_training_loss
from tomato_recon.models.parametric.primitives import generate_plant_geometry
from tomato_recon.export.ply import write_mesh_ply
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.models.graph.edge_head import BiologicalGraphModel
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


def _inverse_softplus(value: float) -> float:
    value = max(value, 1e-6)
    return float(torch.log(torch.expm1(torch.tensor(value))))


def fitted_parameter_tensor(batch, device: torch.device) -> torch.Tensor:
    target = torch.zeros((*batch.node_valid.shape, PARAMETER_WIDTH), device=device)
    for batch_index, sample in enumerate(batch.samples):
        parameters = sample.param_target
        if parameters is None:
            if sample.graph_target is None:
                raise ValueError("Stage 4 requires graph and automatically fitted parameter targets")
            parameters = fit_parametric_targets(
                sample.graph_target, sample.xyz.cpu().numpy(), sample.semantic.cpu().numpy()
            )
        for organ in parameters.organs:
            for slot in organ.source_node_ids:
                if organ.radius_start_m is not None:
                    target[batch_index, slot, 3] = _inverse_softplus(
                        (float(organ.radius_start_m) - 0.0005) / 0.002
                    )
                if organ.radius_end_m is not None:
                    target[batch_index, slot, 4] = _inverse_softplus(
                        (float(organ.radius_end_m) - 0.0005) / 0.002
                    )
                if organ.leaf_width_coeffs is not None:
                    width = torch.as_tensor(organ.leaf_width_coeffs).flatten()
                    target[batch_index, slot, 5] = _inverse_softplus(
                        (float(width[min(1, len(width) - 1)]) - 0.001) / 0.01
                    )
                if organ.bend_coeffs is not None:
                    bend = torch.as_tensor(organ.bend_coeffs).flatten()[:2]
                    target[batch_index, slot, 6 : 6 + len(bend)] = torch.atanh(
                        (bend.to(device) / 0.01).clamp(-0.999, 0.999)
                    )
                if organ.fruit_radii_m is not None:
                    radii = torch.as_tensor(organ.fruit_radii_m).flatten()[:3]
                    for index, radius in enumerate(radii):
                        target[batch_index, slot, 8 + index] = _inverse_softplus(
                            (float(radius) - 0.002) / 0.01
                        )
    return target


def main(argv: list[str] | None = None) -> None:
    cfg, known = load_config("parametric", argv)
    seed_everything(int(cfg.seed), bool(cfg.trainer.deterministic))
    device = select_device(cfg)
    batch = load_training_batch(cfg, device)
    feature_dim = int(cfg.model.graph.hidden_dim)
    encoder = PointEncoder(
        create_backbone_from_config(cfg.model.encoder),
        int(cfg.model.encoder.num_semantic_classes),
    ).to(device)
    encoder_hash = maybe_load_upstream(
        cfg.model.encoder.checkpoint, encoder, batch.samples[0], cfg
    )
    encoder.requires_grad_(False).eval()
    graph_model = BiologicalGraphModel(
        int(cfg.model.encoder.output_dim),
        int(cfg.model.graph.hidden_dim),
        int(cfg.model.graph.knn_candidates),
        float(cfg.model.graph.max_edge_length_m),
    ).to(device)
    graph_hash = maybe_load_upstream(
        cfg.model.graph.checkpoint, graph_model, batch.samples[0], cfg
    )
    graph_model.requires_grad_(False).eval()
    model = ParametricDecoder(feature_dim, int(cfg.model.parametric.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.trainer.learning_rate))
    start_epoch = 0
    if known.resume:
        checkpoint = load_checkpoint(
            known.resume, model, optimizer=optimizer,
            expected_dataset_compatibility=training_dataset_compatibility(
                cfg, batch.samples[0]
            ),
            expected_max_nodes=int(cfg.data.max_nodes)
        )
        start_epoch = int(checkpoint["epoch"]) + 1
    output_dir = stage_output_dir(cfg, "parametric")
    write_run_metadata(cfg, output_dir)
    loader = create_training_loader(cfg)
    best_loss = float("inf")
    metrics = {}
    prediction = None
    epoch_total = (
        start_epoch + 1
        if bool(cfg.trainer.fast_dev_run)
        else max(int(cfg.trainer.max_epochs), start_epoch + 1)
    )
    for epoch in epoch_range(cfg, start_epoch):
        totals = {"loss": 0.0}
        steps = 0
        progress_bar = TrainingProgress("parametric", epoch + 1, epoch_total, len(loader))
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            with torch.no_grad():
                encoded = encoder(
                    batch.xyz, torch.cat([batch.rgb, batch.normals], -1), batch.point_valid
                )
                progress = (epoch + 1) / max(int(cfg.trainer.max_epochs), 1)
                noise_std = 0.0 if progress <= 0.5 else 0.003
                noisy_xyz = batch.node_xyz + noise_std * torch.randn_like(batch.node_xyz) * batch.node_valid[..., None]
                skeleton = SkeletonPrediction(
                    node_xyz=noisy_xyz,
                    parent_flow=batch.parent_flow,
                    existence_logit=torch.where(batch.node_valid, 8.0, -8.0),
                    confidence=batch.node_valid.float(),
                    valid_mask=batch.node_valid,
                )
                graph_scores = graph_model(skeleton, encoded, batch.point_valid)
            prediction = model(graph_scores.nodes.embedding)
            target = fitted_parameter_tensor(batch, device)
            loss = parametric_training_loss(prediction, target, batch.node_valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            steps += 1
            progress_bar.update(totals, steps)
            if bool(cfg.trainer.fast_dev_run):
                break
        metrics = {"loss": totals["loss"] / max(steps, 1)}
        progress_bar.close(metrics)
        if metrics["loss"] <= best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(
                output_dir / "best.ckpt",
                stage="parametric",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                sample=batch.samples[0],
                epoch=epoch,
                metrics=metrics,
                upstream={
                    name: value
                    for name, value in {"encoder": encoder_hash, "graph": graph_hash}.items()
                    if value
                },
            )
    assert prediction is not None
    graph = batch.samples[0].graph_target
    if graph is None:
        raise ValueError("Stage 4 requires graph targets")
    parameters = model.decode_graph(graph, prediction, batch=0)
    geometry = generate_plant_geometry(
        parameters,
        curve_samples=int(cfg.model.parametric.curve_samples),
        radial_segments=int(cfg.model.parametric.radial_segments),
    )
    metrics["mesh_count"] = float(len(geometry.meshes))
    (output_dir / "smoke_parameters.json").write_text(
        json.dumps(parameters.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_mesh_ply(output_dir / "smoke_geometry.ply", geometry.combined_mesh())
    save_checkpoint(
        output_dir / "best.ckpt",
        stage="parametric",
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        sample=batch.samples[0],
        epoch=epoch,
        metrics=metrics,
        upstream={
            name: value
            for name, value in {"encoder": encoder_hash, "graph": graph_hash}.items()
            if value
        },
    )
    write_metrics(output_dir, metrics)
    print(f"parametric checkpoint: {output_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main(sys.argv[1:])
