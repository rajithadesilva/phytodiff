"""Anatomical observed seeds and globally conditioned latent queries."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import EncoderOutput


@dataclass
class Seeds:
    state: Tensor
    features: Tensor
    organ_logits: Tensor
    structure: Tensor  # log(1 + root path / xyz_scale), junction probability
    observed: Tensor


@torch.no_grad()
def diverse_indices(xyz: Tensor, priority: Tensor, count: int, min_distance: float = 0.0) -> Tensor:
    """Confidence-weighted farthest sampling, without selecting an index twice."""
    count = min(count, len(xyz))
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=xyz.device)
    distance = torch.full_like(priority, torch.inf)
    available = torch.ones_like(priority, dtype=torch.bool)
    selected = []
    index = priority.argmax()
    for _ in range(count):
        selected.append(index)
        available[index] = False
        distance = torch.minimum(distance, (xyz - xyz[index]).square().sum(-1))
        available &= distance > min_distance ** 2
        if not available.any():
            break
        index = (distance * (0.25 + priority)).masked_fill(~available, -1).argmax()
    return torch.stack(selected)


class AnatomicalSeeds(nn.Module):
    def __init__(self, point_dim, global_dim, cfg):
        super().__init__()
        self.cfg = cfg
        width = int(cfg.hidden_dim)
        self.observed = nn.Linear(point_dim + 7, width)
        self.queries = nn.Parameter(torch.randn(int(cfg.latent_seeds), width) * 0.02)
        self.global_projection = nn.Linear(global_dim, width)
        self.latent_xyz = nn.Linear(width, 3)
        self.latent_exist = nn.Linear(width, 1)
        self.organ_head = nn.Linear(width, 5)
        self.structure_head = nn.Linear(width, 2)
        self.class_embedding = nn.Embedding(5, width) if cfg.anatomical_embedding else None

    def forward(self, output: EncoderOutput, *, generator=None) -> Seeds:
        xyz, feat = output.point_xyz[0], output.point_features[0]
        semantic = output.semantic_logits[0].softmax(-1)
        skel = output.skeleton_logits[0, :, 0]
        junction = output.junction_logits[0, :, 0].sigmoid()
        shifted = xyz + output.centreline_offset[0]
        indices = diverse_indices(shifted.detach(), (skel.sigmoid() + junction).detach(),
                                  int(self.cfg.observed_seeds), float(self.cfg.min_seed_distance_m))
        obs_features = self.observed(torch.cat((feat, semantic, skel.sigmoid()[:, None],
                                                junction[:, None]), -1)[indices])
        latent = self.queries + self.global_projection(output.global_feature[0])[None]
        centre = (xyz.amin(0) + xyz.amax(0)) * 0.5
        extent = (xyz.amax(0) - xyz.amin(0)).clamp_min(float(self.cfg.xyz_scale_m))
        latent_xyz = centre + self.latent_xyz(latent).tanh() * extent
        noise = torch.randn(latent_xyz.shape, device=xyz.device, dtype=xyz.dtype,
                            generator=generator) * float(self.cfg.latent_noise_m)
        positions = torch.cat((shifted[indices], latent_xyz + noise), 0)
        exist = torch.cat((skel[indices].clamp(-3, 3), self.latent_exist(latent)[:, 0] - 1))
        features = torch.cat((obs_features, latent), 0)
        organs = self.organ_head(features)
        # Stage 1 semantics: background, leaf, main stem, support, side stem.
        # Skeleton organ labels use a different ordering; map explicitly.
        prior = torch.stack((semantic[:, 2], semantic[:, 4], semantic[:, 1],
                             semantic[:, 0] * 0, semantic[:, 0] + semantic[:, 3]), -1)
        prior = prior[indices].clamp_min(1e-5).log()
        organs = organs + torch.cat((prior, torch.zeros_like(organs[len(indices):])), 0)
        structure = self.structure_head(features)
        structure = torch.stack((torch.nn.functional.softplus(structure[:, 0]),
                                 structure[:, 1].sigmoid()), -1)
        if self.class_embedding is not None:
            features = features + organs.softmax(-1) @ self.class_embedding.weight
        observed = torch.arange(len(positions), device=xyz.device) < len(indices)
        return Seeds(torch.cat((positions, exist[:, None]), -1), features, organs,
                     structure, observed)
