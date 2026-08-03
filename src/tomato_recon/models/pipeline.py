from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from tomato_recon.data.schemas import (
    ParametricPlant,
    PlantBatch,
    PlantGeometry,
    PlantGraph,
    SkeletonPrediction,
)
from tomato_recon.models.diffusion.model import ConditionalSkeletonDenoiser
from tomato_recon.models.diffusion.sampling import sample_skeleton
from tomato_recon.models.diffusion.scheduler import DiffusionScheduler
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.models.graph.decode import decode_plant_graph
from tomato_recon.models.graph.edge_head import BiologicalGraphModel
from tomato_recon.models.parametric.decoder import ParametricDecoder
from tomato_recon.models.parametric.primitives import generate_plant_geometry


@dataclass
class PipelineResult:
    skeleton: SkeletonPrediction
    graph: PlantGraph
    parameters: ParametricPlant
    geometry: PlantGeometry
    traits: dict[str, Any]
    uncertainty: dict[str, Any]
    semantic_logits: torch.Tensor


def derive_traits(graph: PlantGraph, geometry: PlantGeometry) -> dict[str, Any]:
    node = {item.id: torch.tensor(item.xyz) for item in graph.nodes}
    lengths = [float(torch.linalg.vector_norm(node[e.child] - node[e.parent])) for e in graph.edges]
    z = torch.tensor([item.xyz[2] for item in graph.nodes])
    return {
        "plant_height_m": float(z.max() - z.min()),
        "total_skeleton_length_m": float(sum(lengths)),
        "mean_internode_length_m": float(sum(lengths) / max(len(lengths), 1)),
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "leaf_structure_count": sum(n.organ_type == "leaf_structure" for n in graph.nodes),
        "side_stem_count": sum(n.organ_type == "side_stem" for n in graph.nodes),
        "fruit_optional_count": sum(n.organ_type == "fruit_optional" for n in graph.nodes),
        "mesh_count": len(geometry.meshes),
    }


class TomatoReconstructionPipeline(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        encoder_cfg = cfg.model.encoder
        backbone = create_backbone_from_config(encoder_cfg)
        self.encoder = PointEncoder(backbone, int(encoder_cfg.num_semantic_classes))
        diffusion_cfg = cfg.model.diffusion
        self.denoiser = ConditionalSkeletonDenoiser(
            max_nodes=int(diffusion_cfg.max_nodes),
            point_feature_dim=int(encoder_cfg.output_dim),
            global_feature_dim=int(encoder_cfg.global_dim),
            hidden_dim=int(diffusion_cfg.hidden_dim),
            layers=int(diffusion_cfg.layers),
            heads=int(diffusion_cfg.heads),
            local_neighbors=int(diffusion_cfg.local_neighbors),
        )
        self.scheduler = DiffusionScheduler(int(diffusion_cfg.train_timesteps))
        graph_cfg = cfg.model.graph
        self.graph_model = BiologicalGraphModel(
            point_feature_dim=int(encoder_cfg.output_dim),
            hidden_dim=int(graph_cfg.hidden_dim),
            knn_candidates=int(graph_cfg.knn_candidates),
            max_edge_length_m=float(graph_cfg.max_edge_length_m),
        )
        self.parametric_decoder = ParametricDecoder(
            node_feature_dim=int(graph_cfg.hidden_dim), hidden_dim=int(cfg.model.parametric.hidden_dim)
        )
        self.cfg = cfg
        self.checkpoint_hashes: dict[str, str] = {}
        self.git_commit = ""

    @torch.no_grad()
    def reconstruct(self, batch: PlantBatch) -> list[PipelineResult]:
        features = torch.cat([batch.rgb, batch.normals], dim=-1)
        encoder_output = self.encoder(batch.xyz, features, batch.point_valid)
        sample_sets: list[SkeletonPrediction] = []
        for sample_index in range(int(self.cfg.inference.num_diffusion_samples)):
            sample_sets.append(
                sample_skeleton(
                    self.denoiser,
                    self.scheduler,
                    encoder_output,
                    batch.point_valid,
                    sample_steps=int(self.cfg.model.diffusion.sample_steps),
                    existence_threshold=float(self.cfg.model.diffusion.existence_threshold),
                    nms_distance_m=float(self.cfg.model.diffusion.nms_distance_m),
                    min_nodes=int(self.cfg.model.diffusion.min_nodes),
                    seed=int(self.cfg.inference.seed) + sample_index,
                    sample_id=sample_index,
                )
            )
        chosen_per_batch = []
        for batch_index in range(len(batch.plant_ids)):
            chosen_per_batch.append(
                max(
                    sample_sets,
                    key=lambda item: float(
                        item.confidence[batch_index][item.valid_mask[batch_index]].mean()
                    ),
                )
            )
        results = []
        for batch_index, (plant_id, skeleton) in enumerate(zip(batch.plant_ids, chosen_per_batch, strict=True)):
            graph_scores = self.graph_model(skeleton, encoder_output, batch.point_valid)
            source = {
                "dataset": batch.samples[batch_index].metadata.get("dataset", "unknown"),
                "preprocessing_hash": batch.samples[batch_index].metadata.get("preprocessing_hash"),
                "checkpoint_hashes": self.checkpoint_hashes,
                "git_commit": self.git_commit,
                "cultivar": batch.samples[batch_index].metadata.get("cultivar"),
                "normalised_to_original": batch.samples[batch_index].metadata.get(
                    "normalised_to_original", []
                ),
            }
            graph = decode_plant_graph(
                plant_id,
                skeleton,
                graph_scores,
                batch=batch_index,
                enforce_botanical_constraints=bool(
                    self.cfg.model.graph.enforce_botanical_constraints
                ),
                allow_fruit=bool(self.cfg.fruit.enabled)
                and bool(
                    batch.samples[batch_index].metadata.get("fruit_pseudo_available", False)
                ),
                source=source,
            )
            raw_parameters = self.parametric_decoder(graph_scores.nodes.embedding)
            parameters = self.parametric_decoder.decode_graph(
                graph, raw_parameters, batch=batch_index
            )
            geometry = generate_plant_geometry(
                parameters,
                curve_samples=int(self.cfg.model.parametric.curve_samples),
                radial_segments=int(self.cfg.model.parametric.radial_segments),
            )
            graph.organs = [
                {
                    "organ_id": organ.organ_id,
                    "organ_type": organ.organ_type,
                    "parent_organ_id": organ.parent_organ_id,
                    "source_node_ids": organ.source_node_ids,
                    "confidence": organ.confidence,
                    "visibility": organ.visibility,
                    "parameter_json_reference": "organ_parameters.json",
                }
                for organ in parameters.organs
            ]
            positions = torch.stack(
                [item.node_xyz[batch_index] for item in sample_sets], dim=0
            )
            node_std = positions.std(dim=0, unbiased=False).norm(dim=-1)
            uncertainty = {
                "diffusion_samples": len(sample_sets),
                "mean_slot_position_std_m": float(node_std.mean().cpu()),
                "max_slot_position_std_m": float(node_std.max().cpu()),
                "mean_retained_confidence": float(
                    skeleton.confidence[batch_index][skeleton.valid_mask[batch_index]].mean().cpu()
                ),
                "node_confidence": [n.existence_confidence for n in graph.nodes],
                "edge_confidence": [e.confidence for e in graph.edges],
                "organ_confidence": [o.confidence for o in parameters.organs],
                "geometry_visibility_weight_map": [
                    {
                        "mesh": mesh.name,
                        "vertex_count": len(mesh.vertices),
                        "weight": float(
                            geometry.visibility_weights[
                                sum(len(previous.vertices) for previous in geometry.meshes[:mesh_index])
                            ].cpu()
                        ),
                    }
                    for mesh_index, mesh in enumerate(geometry.meshes)
                ],
            }
            traits = derive_traits(graph, geometry)
            graph.traits = traits
            results.append(
                PipelineResult(
                    skeleton=skeleton,
                    graph=graph,
                    parameters=parameters,
                    geometry=geometry,
                    traits=traits,
                    uncertainty=uncertainty,
                    semantic_logits=encoder_output.semantic_logits[batch_index],
                )
            )
        return results
