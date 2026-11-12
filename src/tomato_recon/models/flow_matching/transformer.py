"""Time-conditioned query attention, local geometry and soft parent context."""

import math

import torch
from torch import nn


def parent_mask(logits, active=None):
    valid = ~torch.eye(len(logits), dtype=torch.bool, device=logits.device)
    if active is not None:
        valid = valid & active[:, None] & active[None, :]
    return valid


def parent_probabilities(logits, active=None):
    valid = parent_mask(logits, active)
    probabilities = logits.masked_fill(~valid, -1e4).softmax(-1) * valid
    return probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)


class ParentHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)

    def forward(self, features, xyz, *, distance=None):
        logits = self.query(features) @ self.key(features).T / math.sqrt(features.shape[-1])
        if distance is None:
            distance = torch.cdist(xyz.float(), xyz.float())
        return (logits - distance).masked_fill(
            ~parent_mask(logits), -1e4)


class QueryBlock(nn.Module):
    def __init__(self, width, heads, dropout, neighbors):
        super().__init__()
        self.neighbors = neighbors
        self.self_attention = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.local = nn.Sequential(nn.Linear(width + 3, width), nn.GELU(), nn.Linear(width, width))
        self.parent = ParentHead(width)
        self.parent_context = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(width * 4, width), nn.Dropout(dropout))

    def forward(self, h, xyz, memory, existence, *, distance=None, neighbor_indices=None):
        q = self.norms[0](h)[None]
        h = h + self.self_attention(q, q, q, need_weights=False)[0][0]
        h = h + self.cross_attention(self.norms[1](h)[None], memory[None], memory[None],
                                     need_weights=False)[0][0]
        if len(h) > 1:
            if neighbor_indices is None:
                local_distance = torch.cdist(xyz.detach().float(), xyz.detach().float())
                local_distance.fill_diagonal_(torch.inf)
                neighbor_indices = local_distance.topk(
                    min(self.neighbors, len(h) - 1), largest=False).indices
            indices = neighbor_indices
            local = torch.cat((h[indices] - h[:, None], xyz[indices] - xyz[:, None]), -1)
            h = h + self.local(local).mean(1)
        logits = self.parent(self.norms[2](h), xyz, distance=distance)
        probability = parent_probabilities(logits)
        probability = probability * existence.sigmoid()[None]
        probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-8)
        h = h + self.parent_context(probability @ h)
        return h + self.ff(self.norms[3](h))


class QueryTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = int(cfg.hidden_dim)
        self.scale = float(cfg.xyz_scale_m)
        self.existence_scale = float(cfg.existence_scale)
        self.neighbors = int(cfg.graph_neighbors)
        self.state_projection = nn.Linear(4, width)
        self.time = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([QueryBlock(width, int(cfg.heads), float(cfg.dropout),
                                                int(cfg.graph_neighbors))
                                     for _ in range(int(cfg.blocks))])
        self.norm = nn.LayerNorm(width)
        self.velocity = nn.Linear(width, 4)
        self.parent = ParentHead(width)
        self.role = nn.Linear(width, 4)
        self.organ = nn.Linear(width, 5)

    def forward(self, state, time, features, memory, *, velocity_only=False):
        normalized = torch.cat((state[:, :3] / self.scale,
                                state[:, 3:] / self.existence_scale), -1)
        frequencies = torch.exp(torch.linspace(0, math.log(1000), 16, device=state.device))
        phase = time.reshape(-1)[0] * frequencies
        embedding = self.time(torch.cat((phase.sin(), phase.cos())))
        h = self.state_projection(normalized) + features + embedding
        # Positions stay fixed across blocks within this vector-field evaluation.
        # Share geometry without detaching the distances used by parent attention.
        xyz = normalized[:, :3]
        distance = torch.cdist(xyz.float(), xyz.float())
        neighbor_indices = None
        if len(state) > 1:
            local_distance = distance.detach().clone()
            local_distance.fill_diagonal_(torch.inf)
            neighbor_indices = local_distance.topk(
                min(self.neighbors, len(state) - 1), largest=False).indices
        for block in self.blocks:
            h = block(h, xyz, memory, state[:, 3], distance=distance,
                      neighbor_indices=neighbor_indices)
        h = self.norm(h)
        velocity = self.velocity(h) * state.new_tensor([self.scale] * 3 + [self.existence_scale])
        if velocity_only:
            return {"velocity": velocity}
        return {"velocity": velocity, "parent_logits": self.parent(h, xyz, distance=distance),
                "role_logits": self.role(h), "organ_logits": self.organ(h)}
