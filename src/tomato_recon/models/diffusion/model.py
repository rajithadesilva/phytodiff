from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import EncoderOutput


@dataclass
class DenoiserOutput:
    predicted_noise: Tensor
    existence_logit: Tensor
    confidence_logit: Tensor


def sinusoidal_embedding(timestep: Tensor, width: int) -> Tensor:
    half = width // 2
    frequency = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=timestep.device) / max(half - 1, 1)
    )
    phase = timestep.float().unsqueeze(-1) * frequency
    embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
    if width % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def gather_local_point_features(
    node_xyz: Tensor,
    point_xyz: Tensor,
    point_features: Tensor,
    point_mask: Tensor,
    neighbors: int,
) -> Tensor:
    k = min(neighbors, point_xyz.shape[1])
    chunks = []
    batch_index = torch.arange(len(node_xyz), device=node_xyz.device)[:, None, None]
    for start in range(0, node_xyz.shape[1], 64):
        query = node_xyz[:, start : start + 64]
        distances = torch.cdist(query, point_xyz)
        distances = distances.masked_fill(~point_mask[:, None, :], torch.inf)
        indices = distances.topk(k, dim=-1, largest=False).indices
        gathered = point_features[batch_index, indices]
        finite = point_mask[batch_index, indices]
        chunks.append(
            (gathered * finite[..., None]).sum(dim=2)
            / finite.sum(dim=2, keepdim=True).clamp_min(1)
        )
    return torch.cat(chunks, dim=1)


class ConditionalSkeletonDenoiser(nn.Module):
    def __init__(
        self,
        max_nodes: int,
        point_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
        heads: int = 4,
        local_neighbors: int = 16,
    ) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.local_neighbors = local_neighbors
        self.noisy_projection = nn.Linear(6, hidden_dim)
        self.local_projection = nn.Linear(point_feature_dim, hidden_dim)
        self.global_projection = nn.Linear(global_feature_dim, hidden_dim)
        self.time_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.slot_embedding = nn.Parameter(torch.randn(max_nodes, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.noise_head = nn.Linear(hidden_dim, 6)
        self.existence_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        noisy_nodes: Tensor,
        timestep: Tensor,
        encoder_output: EncoderOutput,
        point_mask: Tensor,
    ) -> DenoiserOutput:
        if noisy_nodes.shape[1:] != (self.max_nodes, 6):
            raise ValueError(
                f"noisy_nodes must have shape [B, {self.max_nodes}, 6], got {tuple(noisy_nodes.shape)}"
            )
        local = gather_local_point_features(
            noisy_nodes[..., :3],
            encoder_output.point_xyz,
            encoder_output.point_features,
            point_mask,
            self.local_neighbors,
        )
        time = self.time_projection(sinusoidal_embedding(timestep, self.slot_embedding.shape[-1]))
        value = (
            self.noisy_projection(noisy_nodes)
            + self.local_projection(local)
            + self.global_projection(encoder_output.global_feature)[:, None]
            + time[:, None]
            + self.slot_embedding[None]
        )
        value = self.output_norm(self.transformer(value))
        return DenoiserOutput(
            predicted_noise=self.noise_head(value),
            existence_logit=self.existence_head(value).squeeze(-1),
            confidence_logit=self.confidence_head(value).squeeze(-1),
        )
