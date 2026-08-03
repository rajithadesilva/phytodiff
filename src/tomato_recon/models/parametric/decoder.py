from __future__ import annotations

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import OrganParameters, ParametricPlant, PlantGraph
from tomato_recon.models.parametric.organ_grouping import group_graph_organs

PARAMETER_WIDTH = 11


class ParametricDecoder(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, PARAMETER_WIDTH),
        )

    def forward(self, node_features: Tensor) -> Tensor:
        return self.network(node_features)

    def decode_graph(self, graph: PlantGraph, raw: Tensor, *, batch: int = 0) -> ParametricPlant:
        graph.validate()
        if raw.ndim == 3:
            raw = raw[batch]
        chains, node_to_organ = group_graph_organs(graph)
        node_by_id = {node.id: node for node in graph.nodes}
        parent = {edge.child: edge.parent for edge in graph.edges}
        organs = []
        for organ_id, chain in enumerate(chains):
            slots = [node_by_id[node].source_slot if node_by_id[node].source_slot is not None else node for node in chain]
            prediction = raw[slots].mean(dim=0)
            points = torch.tensor(
                [node_by_id[node].xyz for node in chain], device=raw.device, dtype=raw.dtype
            )
            if len(points) == 1:
                points = torch.cat([points, points + torch.tensor([[0.0, 0.0, 0.005]], device=raw.device)])
            points = points + 0.003 * torch.tanh(prediction[:3])[None]
            radius = torch.nn.functional.softplus(prediction[3:5]) * 0.002 + 0.0005
            width = torch.nn.functional.softplus(prediction[5]) * 0.01 + 0.001
            fruit = torch.nn.functional.softplus(prediction[8:11]) * 0.01 + 0.002
            transform = torch.eye(4, device=raw.device, dtype=raw.dtype)
            transform[:3, 3] = points[0]
            start_parent = parent.get(chain[0])
            organ_type = node_by_id[chain[0]].organ_type
            organs.append(
                OrganParameters(
                    organ_id=organ_id,
                    organ_type=organ_type,
                    parent_organ_id=node_to_organ.get(start_parent),
                    attachment_transform=transform,
                    spline_control_points=points,
                    radius_start_m=radius[0] if organ_type in {"main_stem", "side_stem", "unknown"} else None,
                    radius_end_m=radius[1] if organ_type in {"main_stem", "side_stem", "unknown"} else None,
                    leaf_length_m=float(torch.linalg.vector_norm(points[-1] - points[0]).detach().cpu())
                    if organ_type == "leaf_structure"
                    else None,
                    leaf_width_coeffs=torch.stack([width * 0, width, width * 0])
                    if organ_type == "leaf_structure"
                    else None,
                    bend_coeffs=torch.tanh(prediction[6:8]) * 0.01
                    if organ_type == "leaf_structure"
                    else None,
                    fruit_radii_m=fruit if organ_type == "fruit_optional" else None,
                    confidence=min(node_by_id[node].existence_confidence for node in chain),
                    source_node_ids=chain,
                    visibility=node_by_id[chain[0]].visibility,
                )
            )
        return ParametricPlant(plant_id=graph.plant_id, organs=organs)


def parametric_training_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(prediction.dtype).unsqueeze(-1)
    return (
        torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none") * weights
    ).sum() / weights.sum().clamp_min(1)

