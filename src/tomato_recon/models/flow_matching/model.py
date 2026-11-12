"""Replaceable AG-GFM condition encoder, seed generator and vector field."""

import torch
from torch import nn

from .seeds import AnatomicalSeeds, diverse_indices
from .transformer import QueryTransformer


class AGGFMModel(nn.Module):
    def __init__(self, point_dim, global_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.seeds = AnatomicalSeeds(point_dim, global_dim, cfg)
        self.memory_projection = nn.Linear(point_dim + 3, int(cfg.hidden_dim))
        self.field = QueryTransformer(cfg)

    def condition(self, output, *, generator=None):
        seeds = self.seeds(output, generator=generator)
        xyz, features = output.point_xyz[0], output.point_features[0]
        # Spatial pooling into nearest sampled anchors. All input features contribute.
        indices = diverse_indices(xyz, xyz.new_ones(len(xyz)), int(self.cfg.memory_points))
        with torch.no_grad():
            assignment = torch.cdist(xyz.float(), xyz[indices].float()).argmin(-1)
        pooled = features.new_zeros((len(indices), features.shape[-1]))
        pooled.index_add_(0, assignment, features)
        counts = torch.bincount(assignment, minlength=len(indices)).clamp_min(1)[:, None]
        pooled = pooled / counts
        memory = self.memory_projection(torch.cat((pooled, xyz[indices]/self.cfg.xyz_scale_m), -1))
        return seeds, memory

    def forward(self, state, time, seeds, memory):
        return self.field(state, time, seeds.features, memory)

    def predict_velocity(self, state, time, seeds, memory):
        """Skip final topology/classification heads during intermediate ODE steps."""
        return self.field(state, time, seeds.features, memory, velocity_only=True)["velocity"]
