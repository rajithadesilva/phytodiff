from __future__ import annotations

import torch

from tomato_recon.data.schemas import IGNORE_INDEX, PlantBatch, PlantSample


def collate_plant_samples(samples: list[PlantSample]) -> PlantBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    max_points = max(len(sample.xyz) for sample in samples)

    def pad(value: torch.Tensor, length: int, fill: float | int | bool = 0) -> torch.Tensor:
        if len(value) == length:
            return value
        shape = (length - len(value), *value.shape[1:])
        return torch.cat([value, torch.full(shape, fill, dtype=value.dtype)], dim=0)

    return PlantBatch(
        plant_ids=[sample.plant_id for sample in samples],
        xyz=torch.stack([pad(s.xyz, max_points) for s in samples]),
        rgb=torch.stack([pad(s.rgb, max_points) for s in samples]),
        normals=torch.stack([pad(s.normals, max_points) for s in samples]),
        semantic=torch.stack([pad(s.semantic, max_points, IGNORE_INDEX) for s in samples]),
        instance=torch.stack([pad(s.instance, max_points, -1) for s in samples]),
        point_valid=torch.stack([pad(s.point_valid, max_points, False) for s in samples]),
        node_xyz=torch.stack([s.node_xyz for s in samples]),
        parent_flow=torch.stack([s.parent_flow for s in samples]),
        node_valid=torch.stack([s.node_valid for s in samples]),
        parent_index=torch.stack([s.parent_index for s in samples]),
        organ_type=torch.stack([s.organ_type for s in samples]),
        topology_role=torch.stack([s.topology_role for s in samples]),
        visibility=torch.stack([s.visibility for s in samples]),
        samples=samples,
    )

