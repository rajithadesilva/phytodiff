from __future__ import annotations

import torch
from torch import Tensor

from tomato_recon.data.schemas import EncoderOutput, SkeletonPrediction
from tomato_recon.models.diffusion.model import ConditionalSkeletonDenoiser
from tomato_recon.models.diffusion.scheduler import DiffusionScheduler


def farthest_point_indices(xyz: Tensor, score: Tensor, count: int, mask: Tensor) -> Tensor:
    valid = mask.nonzero(as_tuple=False).flatten()
    if not len(valid):
        raise ValueError("cannot initialise skeleton from an empty point cloud")
    first = valid[score[valid].argmax()]
    chosen = [first]
    minimum_distance = torch.full((len(xyz),), torch.inf, device=xyz.device)
    for _ in range(1, min(count, len(valid))):
        distance = ((xyz - xyz[chosen[-1]]) ** 2).sum(-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        objective = minimum_distance + 0.01 * score
        objective = objective.masked_fill(~mask, -torch.inf)
        objective[torch.stack(chosen)] = -torch.inf
        chosen.append(objective.argmax())
    return torch.stack(chosen)


def initialise_slots(
    encoder_output: EncoderOutput,
    point_mask: Tensor,
    max_nodes: int,
    generator: torch.Generator,
) -> Tensor:
    positions = []
    score = encoder_output.skeleton_logits.squeeze(-1).sigmoid()
    for batch in range(len(score)):
        selected = farthest_point_indices(
            encoder_output.point_xyz[batch], score[batch], max_nodes, point_mask[batch]
        )
        selected_xyz = encoder_output.point_xyz[batch, selected]
        if len(selected_xyz) < max_nodes:
            shortage = max_nodes - len(selected_xyz)
            repeated = selected_xyz[torch.arange(shortage, device=selected_xyz.device) % len(selected_xyz)]
            jitter = torch.randn(
                (shortage, 3), generator=generator, device=selected_xyz.device, dtype=selected_xyz.dtype
            ) * 0.002
            selected_xyz = torch.cat([selected_xyz, repeated + jitter], dim=0)
        positions.append(selected_xyz)
    return torch.stack(positions)


def existence_nms(
    xyz: Tensor,
    probability: Tensor,
    threshold: float,
    distance_m: float,
    min_nodes: int = 2,
) -> Tensor:
    keep = torch.zeros_like(probability, dtype=torch.bool)
    order = torch.argsort(probability, descending=True)
    accepted: list[Tensor] = []
    for index in order:
        if probability[index] < threshold and len(accepted) >= min_nodes:
            break
        if not accepted or torch.linalg.vector_norm(xyz[index] - xyz[torch.stack(accepted)], dim=-1).min() >= distance_m:
            keep[index] = True
            accepted.append(index)
    if keep.sum() < min_nodes:
        keep[order[:min(min_nodes, len(order))]] = True
    return keep


@torch.no_grad()
def sample_skeleton(
    model: ConditionalSkeletonDenoiser,
    scheduler: DiffusionScheduler,
    encoder_output: EncoderOutput,
    point_mask: Tensor,
    *,
    sample_steps: int = 50,
    existence_threshold: float = 0.5,
    nms_distance_m: float = 0.006,
    min_nodes: int = 2,
    seed: int = 42,
    sample_id: int = 0,
) -> SkeletonPrediction:
    device = encoder_output.point_xyz.device
    generator = torch.Generator(device=device).manual_seed(seed)
    position_guess = initialise_slots(encoder_output, point_mask, model.max_nodes, generator)
    clean_guess = torch.cat([position_guess, torch.zeros_like(position_guess)], dim=-1)
    timesteps = scheduler.sampling_timesteps(sample_steps)
    terminal = torch.full((len(position_guess),), timesteps[0], device=device, dtype=torch.long)
    noise = torch.randn(clean_guess.shape, generator=generator, device=device, dtype=clean_guess.dtype)
    state = scheduler.q_sample(clean_guess, terminal, noise)
    output = None
    for index, value in enumerate(timesteps):
        timestep = torch.full((len(state),), value, device=device, dtype=torch.long)
        output = model(state, timestep, encoder_output, point_mask)
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        state = scheduler.ddim_step(output.predicted_noise, timestep, previous, state)
        flow = state[..., 3:]
        norm = flow.norm(dim=-1, keepdim=True)
        state[..., 3:] = torch.where(norm > 1e-6, flow / norm, torch.zeros_like(flow))
        minimum = encoder_output.point_xyz.masked_fill(~point_mask[..., None], torch.inf).amin(dim=1)
        maximum = encoder_output.point_xyz.masked_fill(~point_mask[..., None], -torch.inf).amax(dim=1)
        margin = (maximum - minimum).clamp_min(0.01) * 0.05
        state[..., :3] = torch.maximum(
            torch.minimum(state[..., :3], maximum[:, None] + margin[:, None]),
            minimum[:, None] - margin[:, None],
        )
    assert output is not None
    zero = torch.zeros(len(state), device=device, dtype=torch.long)
    output = model(state, zero, encoder_output, point_mask)
    probability = output.existence_logit.sigmoid()
    valid = torch.stack(
        [
            existence_nms(
                state[b, :, :3], probability[b], existence_threshold, nms_distance_m, min_nodes
            )
            for b in range(len(state))
        ]
    )
    return SkeletonPrediction(
        node_xyz=state[..., :3],
        parent_flow=state[..., 3:],
        existence_logit=output.existence_logit,
        confidence=probability * output.confidence_logit.sigmoid(),
        valid_mask=valid,
        sample_id=torch.full((len(state),), sample_id, device=device, dtype=torch.long),
    )
