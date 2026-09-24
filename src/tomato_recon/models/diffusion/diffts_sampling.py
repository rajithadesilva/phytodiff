"""DiffTS-style residual diffusion sampling and direction-guided skeleton edges."""

from __future__ import annotations

from dataclasses import dataclass, replace

import networkx as nx
import torch
from diffusers import DPMSolverMultistepScheduler
from torch import Tensor

from tomato_recon.models.diffusion.diffts_model import Condition, DiffTSModel


@dataclass
class DiffTSPrediction:
    node_xyz: Tensor
    parent_flow: Tensor
    parent_index: Tensor
    edges: Tensor
    predicted_count: int


def create_scheduler(train_timesteps: int = 1000, beta_start: float = 2e-6,
                     beta_end: float = 2e-4) -> DPMSolverMultistepScheduler:
    return DPMSolverMultistepScheduler(
        num_train_timesteps=train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule="linear",
        algorithm_type="sde-dpmsolver++",
        solver_order=2,
        prediction_type="epsilon",
    )


def q_sample(clean: Tensor, timestep: Tensor, noise: Tensor, scheduler) -> Tensor:
    """DiffTS adds a noised residual to clean XYZ and directions."""
    alpha = scheduler.alphas_cumprod.to(clean.device)[timestep.long()]
    while alpha.ndim < clean.ndim:
        alpha = alpha.unsqueeze(-1)
    return clean + (1 - alpha).sqrt() * noise


def farthest_points(xyz: Tensor, count: int) -> Tensor:
    if len(xyz) < 2:
        raise ValueError("DiffTS needs at least two conditioning points")
    center = xyz.mean(0)
    first = (xyz - center).square().sum(-1).argmax()
    selected = [first]
    nearest = torch.full((len(xyz),), torch.inf, device=xyz.device)
    for _ in range(1, count):
        nearest = torch.minimum(nearest, (xyz - xyz[selected[-1]]).square().sum(-1))
        selected.append(nearest.argmax())
    return xyz[torch.stack(selected)]


def direction_mst(xyz: Tensor, flow: Tensor) -> tuple[Tensor, Tensor]:
    """Connect DiffTS nodes by distance and predicted parent direction."""
    count = len(xyz)
    delta = xyz[:, None] - xyz[None, :]
    distance = delta.norm(dim=-1).clamp_min(1e-6)
    # For candidate child i and parent j, flow_i should point from i toward j.
    toward_parent = -delta / distance[..., None]
    alignment = (toward_parent * torch.nn.functional.normalize(flow, dim=-1)[:, None]).sum(-1)
    directed = distance * (1.25 - 0.75 * alignment)
    symmetric = torch.minimum(directed, directed.T).detach().cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    for i in range(count):
        for j in range(i + 1, count):
            graph.add_edge(i, j, weight=float(symmetric[i, j]))
    tree = nx.minimum_spanning_tree(graph)
    root = int(xyz[:, 2].argmin())
    parent = torch.full((count,), -1, dtype=torch.long, device=xyz.device)
    edges = []
    for child, predecessor in nx.bfs_predecessors(tree, root):
        parent[child] = predecessor
        edges.append((predecessor, child))
    return parent, torch.tensor(edges, dtype=torch.long, device=xyz.device).reshape(-1, 2)


@torch.no_grad()
def sample(model: DiffTSModel, condition: Condition, xyz: Tensor, *,
           steps: int, guidance: float, seed: int, scheduler) -> DiffTSPrediction:
    device = xyz.device
    count = int(condition.count_logits.argmax().item()) + 2
    count = max(2, min(model.max_nodes, count))
    initial_xyz = farthest_points(xyz, count)
    initial = torch.cat((initial_xyz, torch.zeros_like(initial_xyz)), -1)
    solver = create_scheduler(
        scheduler.config.num_train_timesteps,
        scheduler.config.beta_start,
        scheduler.config.beta_end,
    )
    solver.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(initial.shape, device=device, generator=generator)
    residual = q_sample(torch.zeros_like(initial), solver.timesteps[0], noise, solver)
    empty = replace(
        condition,
        sparse=condition.sparse.replace_feature(torch.zeros_like(condition.sparse.features)),
    )
    for timestep in solver.timesteps:
        state = initial + residual
        time = timestep.reshape(1)
        cond_noise = model.denoise(state, time, condition)
        uncond_noise = model.denoise(state, time, empty)
        predicted_noise = uncond_noise + guidance * (cond_noise - uncond_noise)
        residual = solver.step(predicted_noise, timestep, residual, generator=generator).prev_sample
    output = initial + residual
    flow = torch.nn.functional.normalize(output[:, 3:], dim=-1)
    parent, edges = direction_mst(output[:, :3], flow)
    return DiffTSPrediction(output[:, :3], flow, parent, edges, count)


def node_chamfer(predicted: Tensor, target: Tensor) -> Tensor:
    distance = torch.cdist(predicted[None].float(), target[None].float())[0]
    return 0.5 * (distance.min(0).values.mean() + distance.min(1).values.mean())


def _sample_edges(xyz: Tensor, edges: Tensor) -> Tensor:
    a, b = xyz[edges[:, 0]], xyz[edges[:, 1]]
    fractions = torch.linspace(0, 1, 5, device=xyz.device)
    return (a[:, None] * (1 - fractions[None, :, None]) +
            b[:, None] * fractions[None, :, None]).reshape(-1, 3)


def skeleton_chamfer(prediction: DiffTSPrediction, target_xyz: Tensor,
                      target_parent: Tensor) -> Tensor:
    if not len(prediction.edges):
        return node_chamfer(prediction.node_xyz, target_xyz)
    children = (target_parent >= 0).nonzero(as_tuple=False).flatten()
    target_edges = torch.stack((target_parent[children], children), -1)
    return node_chamfer(
        _sample_edges(prediction.node_xyz, prediction.edges),
        _sample_edges(target_xyz, target_edges),
    )
