from __future__ import annotations

import math

import torch
from torch import Tensor

from tomato_recon.data.schemas import MeshData


def fruit_ellipsoid(
    centre: Tensor,
    radii_m: Tensor | list[float],
    *,
    latitude_segments: int = 8,
    longitude_segments: int = 12,
    name: str = "fruit",
    organ_id: int = -1,
) -> MeshData:
    radii = torch.as_tensor(radii_m, device=centre.device, dtype=centre.dtype)
    if radii.shape != (3,) or bool((radii <= 0).any()):
        raise ValueError("fruit radii must contain three positive values")
    vertices = []
    for latitude in range(latitude_segments + 1):
        phi = math.pi * latitude / latitude_segments
        for longitude in range(longitude_segments):
            theta = 2 * math.pi * longitude / longitude_segments
            unit = torch.tensor(
                [math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)],
                device=centre.device,
                dtype=centre.dtype,
            )
            vertices.append(centre + radii * unit)
    vertex_tensor = torch.stack(vertices)
    faces = []
    for latitude in range(latitude_segments):
        for longitude in range(longitude_segments):
            nxt = (longitude + 1) % longitude_segments
            a = latitude * longitude_segments + longitude
            b = latitude * longitude_segments + nxt
            c = (latitude + 1) * longitude_segments + longitude
            d = (latitude + 1) * longitude_segments + nxt
            if latitude > 0:
                faces.append([a, c, b])
            if latitude < latitude_segments - 1:
                faces.append([b, c, d])
    normals = torch.nn.functional.normalize((vertex_tensor - centre) / radii.square(), dim=-1)
    mesh = MeshData(
        name=name,
        vertices=vertex_tensor,
        faces=torch.tensor(faces, device=centre.device, dtype=torch.long),
        normals=normals,
        organ_id=organ_id,
        organ_type="fruit_optional",
    )
    mesh.validate()
    return mesh

