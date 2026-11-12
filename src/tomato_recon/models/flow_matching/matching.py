"""Partial Hungarian OT with learned source-side anatomical descriptors."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def target_structure(xyz: Tensor, parent: Tensor, role: Tensor | None, scale: float) -> Tensor:
    """True root path length and junction indicator, used for supervision only."""
    lengths = xyz.new_zeros(len(xyz))
    # Resolve arbitrary node order; detect corrupt cyclic annotations.
    resolved = parent < 0
    for _ in range(len(xyz)):
        ready = (~resolved) & resolved[parent.clamp_min(0)]
        if not ready.any():
            break
        rows = ready.nonzero().flatten()
        lengths[rows] = lengths[parent[rows]] + (xyz[rows] - xyz[parent[rows]]).norm(dim=-1)
        resolved[rows] = True
    if not resolved.all():
        raise ValueError("target parent topology contains a cycle")
    children = torch.bincount(parent[parent >= 0], minlength=len(xyz))
    junction = (children >= 2).float()
    if role is not None:
        junction = torch.where((role >= 0) & (role < 4), (role == 2).float(), junction)
    return torch.stack((torch.log1p(lengths / scale), junction), -1)


@dataclass
class Match:
    source: Tensor
    target: Tensor
    diagnostics: dict[str, float]


@torch.no_grad()
def match_slots(seeds, data, cfg, scale: float) -> Match:
    xyz, target = seeds.state[:, :3], data.target_xyz
    n, m = len(xyz), len(target)
    position = torch.cdist(xyz.float(), target.float()) / scale
    semantic = torch.zeros_like(position)
    if data.organ_type is not None:
        known = (data.organ_type >= 0) & (data.organ_type < 4)
        semantic[:, known] = -seeds.organ_logits.log_softmax(-1)[:, data.organ_type[known]]
    descriptor = target_structure(target, data.target_parent, data.topology_role, scale)
    structure = torch.cdist(seeds.structure.float(), descriptor.float(), p=1) / 2
    cost = cfg.position * position + cfg.semantic * semantic + cfg.structure * structure
    # n source + m target dummy rows/columns: either side can remain unmatched.
    full = np.full((n + m, n + m), float(cfg.unmatched_cost), dtype=np.float64)
    full[:n, :m] = cost.cpu().numpy()
    full[n:, m:] = 0
    rows, cols = linear_sum_assignment(full)
    keep = (rows < n) & (cols < m)
    source = torch.as_tensor(rows[keep], device=xyz.device, dtype=torch.long)
    dest = torch.as_tensor(cols[keep], device=xyz.device, dtype=torch.long)
    diagnostics = {name: float(value[source, dest].mean()) if len(source) else 0.0
                   for name, value in (("position", position), ("semantic", semantic),
                                       ("structure", structure), ("total", cost))}
    diagnostics.update(matched=float(len(source)), unmatched_slots=float(n - len(source)),
                       unmatched_targets=float(m - len(source)), target_coverage=len(source)/max(m, 1))
    return Match(source, dest, diagnostics)
