from __future__ import annotations

import torch
from torch import Tensor


def skeleton_metrics(
    predicted_xyz: Tensor,
    target_xyz: Tensor,
    *,
    threshold_m: float = 0.01,
    duplicate_distance_m: float = 0.004,
) -> dict[str, float]:
    if not len(predicted_xyz) or not len(target_xyz):
        return {
            "node_chamfer_m2": float("nan"),
            "precision": 0.0,
            "recall": 0.0,
            "coverage": 0.0,
            "duplicate_rate": 0.0,
        }
    distance = torch.cdist(predicted_xyz, target_xyz)
    pred_nearest = distance.min(dim=1).values
    target_nearest = distance.min(dim=0).values
    pairwise = torch.cdist(predicted_xyz, predicted_xyz)
    pairwise.fill_diagonal_(torch.inf)
    return {
        "node_chamfer_m2": float(pred_nearest.square().mean() + target_nearest.square().mean()),
        "precision": float((pred_nearest <= threshold_m).float().mean()),
        "recall": float((target_nearest <= threshold_m).float().mean()),
        "coverage": float((target_nearest <= threshold_m).float().mean()),
        "duplicate_rate": float((pairwise.min(dim=1).values < duplicate_distance_m).float().mean()),
    }


def binary_precision_recall(logit: Tensor, target: Tensor, threshold: float = 0.5) -> dict[str, float]:
    predicted = logit.sigmoid() >= threshold
    target = target.bool()
    true_positive = (predicted & target).sum()
    return {
        "precision": float(true_positive / (predicted.sum().clamp_min(1))),
        "recall": float(true_positive / (target.sum().clamp_min(1))),
    }

