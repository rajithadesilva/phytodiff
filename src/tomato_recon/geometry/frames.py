from __future__ import annotations

import torch
from torch import Tensor


def _safe_normalise(value: Tensor, fallback: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(value)
    return value / norm if float(norm) > 1e-8 else fallback


def parallel_transport_frames(curve: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if curve.ndim != 2 or curve.shape[-1] != 3 or len(curve) < 2:
        raise ValueError("curve must have shape [S, 3] with S >= 2")
    delta = torch.empty_like(curve)
    delta[0] = curve[1] - curve[0]
    delta[-1] = curve[-1] - curve[-2]
    if len(curve) > 2:
        delta[1:-1] = curve[2:] - curve[:-2]
    tangent = torch.nn.functional.normalize(delta, dim=-1)
    axes = torch.eye(3, device=curve.device, dtype=curve.dtype)
    initial_axis = axes[torch.argmin(torch.abs(axes @ tangent[0]))]
    first_normal = torch.nn.functional.normalize(torch.cross(tangent[0], initial_axis, dim=-1), dim=-1)
    normals = [first_normal]
    for index in range(1, len(curve)):
        projected = normals[-1] - torch.dot(normals[-1], tangent[index]) * tangent[index]
        if float(torch.linalg.vector_norm(projected)) < 1e-8:
            axis = axes[torch.argmin(torch.abs(axes @ tangent[index]))]
            projected = torch.cross(tangent[index], axis, dim=-1)
        normals.append(torch.nn.functional.normalize(projected, dim=-1))
    normal = torch.stack(normals)
    binormal = torch.nn.functional.normalize(torch.cross(tangent, normal, dim=-1), dim=-1)
    return tangent, normal, binormal

