from __future__ import annotations

import torch
from torch import Tensor


def occlusion_aware_geometry_loss(
    scan_points: Tensor, model_points: Tensor, visibility_weight: Tensor
) -> Tensor:
    distance = torch.cdist(scan_points, model_points).square()
    scan_to_model = distance.min(dim=1).values.mean()
    model_to_scan = (distance.min(dim=0).values * visibility_weight).sum() / visibility_weight.sum().clamp_min(1e-8)
    return scan_to_model + model_to_scan


def normal_consistency_loss(
    scan_points: Tensor,
    scan_normals: Tensor,
    model_points: Tensor,
    model_normals: Tensor,
    visibility_weight: Tensor,
) -> Tensor:
    """Nearest-surface normal loss, ignoring unavailable scan normals."""
    nearest_scan = torch.cdist(model_points, scan_points).argmin(dim=1)
    reliable = scan_normals[nearest_scan].norm(dim=-1) > 1e-6
    reliable = reliable & (model_normals.norm(dim=-1) > 1e-6)
    if not reliable.any():
        return model_points.sum() * 0
    scan_unit = torch.nn.functional.normalize(scan_normals[nearest_scan[reliable]], dim=-1)
    model_unit = torch.nn.functional.normalize(model_normals[reliable], dim=-1)
    disagreement = 1 - (scan_unit * model_unit).sum(dim=-1).abs()
    weight = visibility_weight[reliable]
    return (disagreement * weight).sum() / weight.sum().clamp_min(1e-8)


def positive_radius_regularisation(radius: Tensor) -> Tensor:
    return torch.relu(-radius).mean() + 0.01 * radius.square().mean()
