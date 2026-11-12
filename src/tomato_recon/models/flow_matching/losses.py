"""Masked flow, endpoint, parent, anatomical and geometric supervision."""

import torch
from torch.nn import functional as F

from .flow import interpolate, supervised_endpoint
from .matching import match_slots, target_structure
from .transformer import parent_mask


def parent_targets(count, data, match):
    result = torch.full((count,), -100, dtype=torch.long, device=match.source.device)
    inverse = torch.full((len(data.target_xyz),), -1, dtype=torch.long, device=result.device)
    inverse[match.target] = match.source
    target_parent = data.target_parent[match.target]
    rows = (target_parent >= 0)
    source, parent = match.source[rows], inverse[target_parent[rows]]
    valid = (parent >= 0) & (parent != source)
    result[source[valid]] = parent[valid]
    return result


def chamfer(a, b):
    if not len(a) or not len(b):
        return a.sum() * 0
    d = torch.cdist(a.float(), b.float())
    return (d.min(0).values.mean() + d.min(1).values.mean()) / 2


def training_loss(model, seeds, memory, data, cfg, *, time=None):
    match = match_slots(seeds, data, cfg.matching, float(cfg.xyz_scale_m))
    endpoint = supervised_endpoint(seeds, data, match, cfg)
    time = torch.rand((), device=endpoint.device) if time is None else endpoint.new_tensor(time)
    state, target_velocity = interpolate(seeds.state, endpoint, time)
    output = model(state, time, seeds, memory)
    predicted_endpoint = state + (1 - time) * output["velocity"]
    scale = state.new_tensor([float(cfg.xyz_scale_m)] * 3 + [float(cfg.existence_scale)])
    flow = F.mse_loss(output["velocity"] / scale, target_velocity / scale)
    active = torch.zeros(len(state), dtype=torch.bool, device=state.device)
    active[match.source] = True
    exist = F.binary_cross_entropy_with_logits(predicted_endpoint[:, 3], active.float())
    # Parent/role heads are time-conditioned and supervised at the sampled t.
    targets = parent_targets(len(state), data, match)
    mask = parent_mask(output["parent_logits"], active)
    logits = output["parent_logits"].masked_fill(~mask, -1e4)
    valid_parent = targets >= 0
    zero = output["velocity"].sum() * 0
    parent = F.cross_entropy(logits[valid_parent], targets[valid_parent]) if valid_parent.any() else zero
    role = zero
    if data.topology_role is not None:
        labels = data.topology_role[match.target]
        valid = (labels >= 0) & (labels < 4)
        if valid.any():
            role = F.cross_entropy(output["role_logits"][match.source[valid]], labels[valid])
    geom, structure, organ = zero, zero, zero
    if len(match.source):
        # Only matched slots enter geometry. Unmatched GT remains represented in
        # Chamfer, so partial OT cannot hide uncovered branches in the diagnostics.
        geom = chamfer(predicted_endpoint[match.source, :3], data.target_xyz) / cfg.xyz_scale_m
        descriptor = target_structure(data.target_xyz, data.target_parent, data.topology_role,
                                       float(cfg.xyz_scale_m))
        structure = F.smooth_l1_loss(seeds.structure[match.source], descriptor[match.target])
        if data.organ_type is not None:
            labels = data.organ_type[match.target]
            known = (labels >= 0) & (labels < 4)
            if known.any():
                organ = F.cross_entropy(seeds.organ_logits[match.source[known]], labels[known])
                organ = organ + F.cross_entropy(output["organ_logits"][match.source[known]], labels[known])
    geom = geom + cfg.loss.structure * structure + cfg.loss.organ * organ
    losses = dict(flow=flow, exist=exist, parent=parent, role=role, geom=geom)
    total = sum(float(cfg.loss[name]) * value for name, value in losses.items())
    return {"loss": total, **losses, "structure": structure, "organ": organ}, match.diagnostics
