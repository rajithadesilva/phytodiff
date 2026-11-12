from __future__ import annotations

import torch
from torch import Tensor

from tomato_recon.data.schemas import MeshData


def sample_mesh_surface(mesh: MeshData, samples: int, seed: int = 0) -> Tensor:
    mesh.validate()
    triangles = mesh.vertices[mesh.faces]
    cross = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
    area = torch.linalg.vector_norm(cross, dim=-1) * 0.5
    if float(area.sum()) <= 0:
        raise ValueError("cannot sample a zero-area mesh")
    generator = torch.Generator(device=mesh.vertices.device).manual_seed(seed)
    face_index = torch.multinomial(area, samples, replacement=True, generator=generator)
    selected = triangles[face_index]
    uv = torch.rand((samples, 2), generator=generator, device=mesh.vertices.device)
    root = torch.sqrt(uv[:, :1])
    weights = torch.cat([1 - root, root * (1 - uv[:, 1:]), root * uv[:, 1:]], dim=-1)
    return (selected * weights[..., None]).sum(dim=1)


def chamfer_distance(first: Tensor, second: Tensor) -> Tensor:
    if not len(first) or not len(second):
        raise ValueError("Chamfer distance requires non-empty point sets")
    distance = torch.cdist(first, second).square()
    return distance.min(dim=1).values.mean() + distance.min(dim=0).values.mean()

