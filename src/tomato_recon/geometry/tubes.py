from __future__ import annotations

import math

import torch
from torch import Tensor

from tomato_recon.data.schemas import MeshData
from tomato_recon.geometry.frames import parallel_transport_frames
from tomato_recon.geometry.spline import sample_cubic_bspline


def stem_tube(
    control_points: Tensor,
    radius_start_m: float | Tensor,
    radius_end_m: float | Tensor,
    *,
    curve_samples: int = 16,
    radial_segments: int = 10,
    name: str = "stem",
    organ_id: int = -1,
    organ_type: str = "main_stem",
) -> MeshData:
    if radial_segments < 3:
        raise ValueError("radial_segments must be at least three")
    radius_start = torch.as_tensor(radius_start_m, device=control_points.device, dtype=control_points.dtype)
    radius_end = torch.as_tensor(radius_end_m, device=control_points.device, dtype=control_points.dtype)
    if float(radius_start) <= 0 or float(radius_end) <= 0:
        raise ValueError("stem radii must be positive")
    curve = sample_cubic_bspline(control_points, curve_samples)
    _, normal, binormal = parallel_transport_frames(curve)
    angle = torch.arange(radial_segments, device=curve.device, dtype=curve.dtype) * (
        2 * math.pi / radial_segments
    )
    radial = torch.cos(angle)[None, :, None] * normal[:, None] + torch.sin(angle)[
        None, :, None
    ] * binormal[:, None]
    ratio = torch.linspace(0, 1, curve_samples, device=curve.device, dtype=curve.dtype)
    radius = radius_start * (1 - ratio) + radius_end * ratio
    vertices = (curve[:, None] + radius[:, None, None] * radial).reshape(-1, 3)
    normals = radial.reshape(-1, 3)
    faces = []
    for row in range(curve_samples - 1):
        for column in range(radial_segments):
            nxt = (column + 1) % radial_segments
            a = row * radial_segments + column
            b = row * radial_segments + nxt
            c = (row + 1) * radial_segments + column
            d = (row + 1) * radial_segments + nxt
            faces.extend([[a, c, b], [b, c, d]])
    mesh = MeshData(
        name=name,
        vertices=vertices,
        faces=torch.tensor(faces, device=curve.device, dtype=torch.long),
        normals=normals,
        organ_id=organ_id,
        organ_type=organ_type,
    )
    mesh.validate()
    return mesh

