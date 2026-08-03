from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import EncoderOutput, SkeletonPrediction


@dataclass
class NodeScores:
    embedding: Tensor
    organ_type_logits: Tensor
    topology_role_logits: Tensor
    visibility_logits: Tensor
    root_logit: Tensor


def interpolate_nearest(
    node_xyz: Tensor, point_xyz: Tensor, point_features: Tensor, point_mask: Tensor
) -> Tensor:
    chunks = []
    for start in range(0, node_xyz.shape[1], 64):
        distances = torch.cdist(node_xyz[:, start : start + 64], point_xyz)
        nearest = distances.masked_fill(~point_mask[:, None], torch.inf).argmin(dim=-1)
        chunks.append(
            torch.gather(
                point_features,
                1,
                nearest[..., None].expand(-1, -1, point_features.shape[-1]),
            )
        )
    return torch.cat(chunks, dim=1)


def local_distance_statistics(
    node_xyz: Tensor, point_xyz: Tensor, point_mask: Tensor, neighbors: int = 4
) -> tuple[Tensor, Tensor]:
    means, deviations = [], []
    k = min(neighbors, point_xyz.shape[1])
    for start in range(0, node_xyz.shape[1], 64):
        distances = torch.cdist(node_xyz[:, start : start + 64], point_xyz)
        distances = distances.masked_fill(~point_mask[:, None], torch.inf)
        nearest = distances.topk(k, largest=False).values
        means.append(nearest.mean(-1))
        deviations.append(nearest.std(-1, unbiased=False))
    return torch.cat(means, dim=1), torch.cat(deviations, dim=1)


class NodeClassificationHead(nn.Module):
    def __init__(self, point_feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = point_feature_dim + 3 + 3 + 1 + 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.organ = nn.Linear(hidden_dim, 5)
        self.role = nn.Linear(hidden_dim, 4)
        self.visibility = nn.Linear(hidden_dim, 3)
        self.root = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        skeleton: SkeletonPrediction,
        encoder_output: EncoderOutput,
        point_mask: Tensor,
    ) -> NodeScores:
        local = interpolate_nearest(
            skeleton.node_xyz,
            encoder_output.point_xyz,
            encoder_output.point_features,
            point_mask,
        )
        distance_mean, distance_std = local_distance_statistics(
            skeleton.node_xyz, encoder_output.point_xyz, point_mask
        )
        local_stats = torch.stack(
            [
                distance_mean,
                distance_std,
                skeleton.node_xyz[..., 2],
                skeleton.parent_flow[..., 2],
            ],
            dim=-1,
        )
        value = torch.cat(
            [
                local,
                skeleton.node_xyz,
                skeleton.parent_flow,
                skeleton.existence_logit.sigmoid().unsqueeze(-1),
                local_stats,
            ],
            dim=-1,
        )
        embedding = self.encoder(value)
        return NodeScores(
            embedding=embedding,
            organ_type_logits=self.organ(embedding),
            topology_role_logits=self.role(embedding),
            visibility_logits=self.visibility(embedding),
            root_logit=self.root(embedding).squeeze(-1),
        )
