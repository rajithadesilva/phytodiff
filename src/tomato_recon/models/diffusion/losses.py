from __future__ import annotations

import torch
from torch import Tensor

from tomato_recon.data.schemas import EncoderOutput
from tomato_recon.models.diffusion.model import ConditionalSkeletonDenoiser, DenoiserOutput
from tomato_recon.models.diffusion.scheduler import DiffusionScheduler


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(prediction.dtype)
    while weights.ndim < prediction.ndim:
        weights = weights.unsqueeze(-1)
    return ((prediction - target).square() * weights).sum() / weights.sum().clamp_min(1)


def focal_bce(logit: Tensor, target: Tensor, gamma: float = 2.0) -> Tensor:
    target = target.to(logit.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    probability = torch.where(target > 0.5, logit.sigmoid(), 1 - logit.sigmoid())
    return ((1 - probability).pow(gamma) * bce).mean()


def duplicate_penalty(node_xyz: Tensor, existence_logit: Tensor, sigma: float) -> Tensor:
    distance2 = torch.cdist(node_xyz, node_xyz).square()
    probability = existence_logit.sigmoid()
    pair_weight = probability[:, :, None] * probability[:, None, :]
    upper = torch.triu(torch.ones_like(distance2, dtype=torch.bool), diagonal=1)
    values = torch.exp(-distance2 / max(sigma**2, 1e-12)) * pair_weight
    return values[upper].mean() if upper.any() else values.sum() * 0


def bounds_penalty(node_xyz: Tensor, existence_logit: Tensor, point_xyz: Tensor, point_mask: Tensor) -> Tensor:
    minimum = point_xyz.masked_fill(~point_mask[..., None], torch.inf).amin(dim=1)
    maximum = point_xyz.masked_fill(~point_mask[..., None], -torch.inf).amax(dim=1)
    margin = (maximum - minimum).clamp_min(0.01) * 0.05
    below = torch.relu(minimum[:, None] - margin[:, None] - node_xyz)
    above = torch.relu(node_xyz - maximum[:, None] - margin[:, None])
    return ((below + above).square().sum(-1) * existence_logit.sigmoid()).mean()


def diffusion_training_loss(
    model: ConditionalSkeletonDenoiser,
    scheduler: DiffusionScheduler,
    encoder_output: EncoderOutput,
    point_mask: Tensor,
    node_xyz: Tensor,
    parent_flow: Tensor,
    node_valid: Tensor,
    timestep: Tensor,
    *,
    visibility: Tensor | None = None,
    duplicate_sigma: float = 0.006,
    existence_weight: float = 1.0,
    confidence_weight: float = 0.25,
    flow_weight: float = 0.25,
    duplicate_weight: float = 0.1,
    bounds_weight: float = 0.05,
) -> tuple[Tensor, dict[str, Tensor], DenoiserOutput]:
    clean = torch.cat([node_xyz, parent_flow], dim=-1)
    noise = torch.randn_like(clean)
    noisy = scheduler.q_sample(clean, timestep, noise)
    output = model(noisy, timestep, encoder_output, point_mask)
    diffusion = masked_mse(output.predicted_noise, noise, node_valid)
    existence = focal_bce(output.existence_logit, node_valid)
    confidence_target = node_valid.to(output.confidence_logit.dtype)
    if visibility is not None:
        visibility_confidence = torch.where(
            visibility == 0,
            torch.ones_like(confidence_target),
            torch.where(
                visibility == 1,
                torch.full_like(confidence_target, 0.6),
                torch.full_like(confidence_target, 0.3),
            ),
        )
        confidence_target = confidence_target * visibility_confidence
    confidence = torch.nn.functional.binary_cross_entropy_with_logits(
        output.confidence_logit, confidence_target
    )
    clean_prediction = scheduler.predict_clean(noisy, output.predicted_noise, timestep)
    predicted_flow = torch.nn.functional.normalize(clean_prediction[..., 3:], dim=-1)
    cosine = 1 - torch.nn.functional.cosine_similarity(predicted_flow, parent_flow, dim=-1)
    non_root = node_valid & (parent_flow.norm(dim=-1) > 0)
    flow = cosine[non_root].mean() if non_root.any() else cosine.sum() * 0
    duplicate = duplicate_penalty(clean_prediction[..., :3], output.existence_logit, duplicate_sigma)
    bounds = bounds_penalty(
        clean_prediction[..., :3], output.existence_logit, encoder_output.point_xyz, point_mask
    )
    total = (
        diffusion
        + existence_weight * existence
        + confidence_weight * confidence
        + flow_weight * flow
        + duplicate_weight * duplicate
        + bounds_weight * bounds
    )
    return total, {
        "diffusion": diffusion,
        "existence": existence,
        "confidence": confidence,
        "flow": flow,
        "duplicate": duplicate,
        "bounds": bounds,
    }, output
