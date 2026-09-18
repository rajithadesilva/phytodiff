from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import EncoderOutput, FeatureMap


class PointBackbone(nn.Module, ABC):
    output_dim: int
    global_dim: int

    @abstractmethod
    def forward(
        self, xyz: Tensor, features: Tensor, mask: Tensor
    ) -> tuple[Tensor, list[FeatureMap], Tensor]:
        """Return per-input-point features, an optional pyramid, and a global feature."""


class PointEncoder(nn.Module):
    def __init__(self, backbone: PointBackbone, num_semantic_classes: int = 5) -> None:
        super().__init__()
        self.backbone = backbone
        width = backbone.output_dim
        self.semantic_head = nn.Linear(width, num_semantic_classes)
        self.skeleton_head = nn.Linear(width, 1)
        self.offset_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.junction_head = nn.Linear(width, 1)

    def forward(self, xyz: Tensor, features: Tensor, mask: Tensor) -> EncoderOutput:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape [B, N, 3]")
        if features.shape[:2] != xyz.shape[:2] or mask.shape != xyz.shape[:2]:
            raise ValueError("encoder feature/mask shapes do not match xyz")
        point_features, pyramid, global_feature = self.backbone(xyz, features, mask)
        return EncoderOutput(
            point_xyz=xyz,
            point_features=point_features,
            multiscale_features=pyramid,
            global_feature=global_feature,
            semantic_logits=self.semantic_head(point_features),
            skeleton_logits=self.skeleton_head(point_features),
            centreline_offset=self.offset_head(point_features),
            junction_logits=self.junction_head(point_features),
        )


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def nearest_skeleton_targets(
    point_xyz: Tensor, node_xyz: Tensor, node_valid: Tensor, threshold_m: float
) -> tuple[Tensor, Tensor]:
    distance = torch.cdist(point_xyz, node_xyz)
    distance = distance.masked_fill(~node_valid[:, None, :], float("inf"))
    nearest_distance, nearest_index = distance.min(dim=-1)
    nearest = torch.gather(node_xyz, 1, nearest_index[..., None].expand(-1, -1, 3))
    skeleton = nearest_distance <= threshold_m
    offset = nearest - point_xyz
    return skeleton, offset


def encoder_losses(
    output: EncoderOutput,
    semantic: Tensor,
    point_valid: Tensor,
    node_xyz: Tensor,
    node_valid: Tensor,
    topology_role: Tensor,
    *,
    skeleton_threshold_m: float,
    junction_threshold_multiplier: float,
    ignore_index: int = -100,
) -> dict[str, Tensor]:
    semantic_target = semantic.masked_fill(~point_valid, ignore_index)
    semantic_logits = output.semantic_logits
    sem = torch.nn.functional.cross_entropy(
        semantic_logits.reshape(-1, semantic_logits.shape[-1]),
        semantic_target.reshape(-1),
        ignore_index=ignore_index,
    )
    skeleton_target, offset_target = nearest_skeleton_targets(
        output.point_xyz, node_xyz, node_valid, skeleton_threshold_m
    )
    valid_float = point_valid.to(output.skeleton_logits.dtype)
    skeleton_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        output.skeleton_logits.squeeze(-1), skeleton_target.float(), reduction="none"
    )
    skeleton_bce = (skeleton_bce * valid_float).sum() / valid_float.sum().clamp_min(1)
    probability = output.skeleton_logits.squeeze(-1).sigmoid() * valid_float
    intersection = (probability * skeleton_target.float()).sum()
    dice = 1 - (2 * intersection + 1) / (
        probability.sum() + skeleton_target.float().sum() + 1
    )
    positive = skeleton_target & point_valid
    offset = torch.nn.functional.smooth_l1_loss(
        output.centreline_offset[positive], offset_target[positive]
    ) if positive.any() else output.centreline_offset.sum() * 0
    junction_nodes = node_xyz.masked_fill(
        ~(node_valid & (topology_role == 2))[..., None], float("nan")
    )
    # Point is a junction target when it lies close to any valid junction node.
    junction_valid = node_valid & (topology_role == 2)
    if junction_valid.any():
        distances = torch.cdist(output.point_xyz, torch.nan_to_num(junction_nodes))
        distances = distances.masked_fill(~junction_valid[:, None, :], float("inf"))
        junction_target = (
            distances.min(dim=-1).values
            <= junction_threshold_multiplier * skeleton_threshold_m
        )
    else:
        junction_target = torch.zeros_like(point_valid)
    junction = torch.nn.functional.binary_cross_entropy_with_logits(
        output.junction_logits.squeeze(-1), junction_target.float(), reduction="none"
    )
    junction = (junction * valid_float).sum() / valid_float.sum().clamp_min(1)
    total = sem + skeleton_bce + dice + offset + 0.5 * junction
    return {
        "loss": total,
        "semantic": sem,
        "skeleton": skeleton_bce + dice,
        "offset": offset,
        "junction": junction,
    }
