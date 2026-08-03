from __future__ import annotations

import math

import torch
from torch import Tensor

from tomato_recon.data.schemas import MeshData
from tomato_recon.geometry.frames import parallel_transport_frames
from tomato_recon.geometry.spline import sample_cubic_bspline


def leaf_surface(
    control_points: Tensor,
    width_coeffs: Tensor | list[float],
    bend_coeffs: Tensor | list[float] | None = None,
    *,
    curve_samples: int = 16,
    name: str = "leaf",
    organ_id: int = -1,
) -> MeshData:
    curve = sample_cubic_bspline(control_points, curve_samples)
    _, normal, binormal = parallel_transport_frames(curve)
    t = torch.linspace(0, 1, curve_samples, device=curve.device, dtype=curve.dtype)
    coeffs = torch.as_tensor(width_coeffs, device=curve.device, dtype=curve.dtype).flatten()
    padded = torch.nn.functional.pad(coeffs[:3], (0, max(0, 3 - len(coeffs))))
    width = padded[0] + padded[1] * torch.sin(math.pi * t) + padded[2] * torch.sin(2 * math.pi * t)
    width = width.abs().clamp_min(1e-5)
    width[0] = 0
    width[-1] = 0
    if bend_coeffs is not None:
        bend = torch.as_tensor(bend_coeffs, device=curve.device, dtype=curve.dtype).flatten()
        bend = torch.nn.functional.pad(bend[:2], (0, max(0, 2 - len(bend))))
        curve = curve + (
            bend[0] * torch.sin(math.pi * t) + bend[1] * torch.sin(2 * math.pi * t)
        )[:, None] * binormal
    left = curve - width[:, None] * normal
    right = curve + width[:, None] * normal
    vertices = torch.stack([left, right], dim=1).reshape(-1, 3)
    faces = []
    for row in range(curve_samples - 1):
        a, b = 2 * row, 2 * row + 1
        c, d = 2 * (row + 1), 2 * (row + 1) + 1
        faces.extend([[a, c, b], [b, c, d]])
    face_tensor = torch.tensor(faces, device=curve.device, dtype=torch.long)
    vertex_normals = torch.zeros_like(vertices)
    triangles = vertices[face_tensor]
    face_normals = torch.nn.functional.normalize(
        torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1),
        dim=-1,
    )
    for corner in range(3):
        vertex_normals.index_add_(0, face_tensor[:, corner], face_normals)
    vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)
    mesh = MeshData(
        name=name,
        vertices=vertices,
        faces=face_tensor,
        normals=vertex_normals,
        organ_id=organ_id,
        organ_type="leaf_structure",
    )
    mesh.validate()
    return mesh

