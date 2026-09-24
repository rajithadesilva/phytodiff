"""Stage 2 view inputs. Inference deliberately never opens a target file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tomato_recon.data.schemas import PlantSample
from tomato_recon.data.top_down import ALGORITHM_VERSION


@dataclass
class DiffTSInput:
    plant_id: str
    dataset: str
    view: str
    xyz: Tensor
    rgb: Tensor
    normals: Tensor
    rgb_available: bool
    target_xyz: Tensor | None = None
    target_flow: Tensor | None = None
    target_parent: Tensor | None = None

    def to(self, device: torch.device | str) -> DiffTSInput:
        return DiffTSInput(
            self.plant_id, self.dataset, self.view,
            self.xyz.to(device), self.rgb.to(device), self.normals.to(device),
            self.rgb_available,
            None if self.target_xyz is None else self.target_xyz.to(device),
            None if self.target_flow is None else self.target_flow.to(device),
            None if self.target_parent is None else self.target_parent.to(device),
        )


def rgb_available(metadata: dict) -> bool:
    """Older TomatoWUR caches predate the explicit RGB availability field."""
    value = metadata.get("rgb_available")
    if value is not None:
        return bool(value)
    dataset = str(metadata.get("dataset", ""))
    if dataset in {"tomatowur", "tomatopgt"}:
        return True
    if dataset in {"pheno4d", "synthetic-test-fixture"}:
        return False
    raise ValueError(f"RGB availability is unknown for dataset {dataset!r}")


def from_sample(sample: PlantSample) -> DiffTSInput:
    points = sample.point_valid.bool()
    nodes = sample.node_valid.bool()
    if points.sum() < 2 or nodes.sum() < 2:
        raise ValueError(f"{sample.plant_id}: stage 2 requires at least two points and nodes")
    node_rows = nodes.nonzero(as_tuple=False).flatten()
    remap = torch.full((len(nodes),), -1, dtype=torch.long)
    remap[node_rows] = torch.arange(len(node_rows))
    original_parent = sample.parent_index[nodes].long()
    target_parent = torch.where(original_parent >= 0, remap[original_parent.clamp_min(0)], -1)
    return DiffTSInput(
        plant_id=sample.plant_id,
        dataset=str(sample.metadata.get("dataset", "unknown")),
        view=str(sample.metadata.get("pcl_type", "full")),
        xyz=sample.xyz[points].float(),
        rgb=sample.rgb[points].float(),
        normals=sample.normals[points].float(),
        rgb_available=rgb_available(sample.metadata),
        target_xyz=sample.node_xyz[nodes].float(),
        target_flow=sample.parent_flow[nodes].float(),
        target_parent=target_parent,
    )


def load_top_down_input(path: str | Path) -> DiffTSInput:
    path = Path(path)
    if path.name != "top_down.npz":
        raise ValueError("stage 2 inference accepts saved top_down.npz views only")
    with np.load(path, allow_pickle=False) as cloud:
        required = {"plant_id", "xyz", "rgb", "normals", "point_valid", "metadata_json"}
        if not required.issubset(cloud.files):
            raise ValueError(f"{path}: missing top-down point fields")
        metadata = json.loads(str(cloud["metadata_json"].item()))
        if metadata.get("top_down", {}).get("algorithm_version") != ALGORITHM_VERSION:
            raise ValueError(f"{path}: stale or invalid top-down view")
        valid = cloud["point_valid"].astype(bool)
        xyz = torch.from_numpy(cloud["xyz"][valid].copy()).float()
        rgb = torch.from_numpy(cloud["rgb"][valid].copy()).float()
        normals = torch.from_numpy(cloud["normals"][valid].copy()).float()
        plant_id = str(cloud["plant_id"].item())
    if len(xyz) < 2 or not torch.isfinite(xyz).all():
        raise ValueError(f"{path}: too few or non-finite top-down points")
    return DiffTSInput(
        plant_id, str(metadata.get("dataset", "unknown")), "top_down",
        xyz, rgb, normals, rgb_available(metadata),
    )


def sample_points(data: DiffTSInput, limit: int, *, seed: int) -> DiffTSInput:
    if len(data.xyz) <= limit:
        return data
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(data.xyz), generator=generator)[:limit]
    return DiffTSInput(
        data.plant_id, data.dataset, data.view,
        data.xyz[indices], data.rgb[indices], data.normals[indices],
        data.rgb_available, data.target_xyz, data.target_flow, data.target_parent,
    )


def augment_z(data: DiffTSInput, *, generator: torch.Generator) -> DiffTSInput:
    """Apply one rigid Z rotation and mild scale to scan and skeleton together."""
    angle = float(torch.rand((), generator=generator) * (2 * torch.pi))
    scale = float(0.9 + 0.2 * torch.rand((), generator=generator))
    c, s = np.cos(angle), np.sin(angle)
    rotation = data.xyz.new_tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return DiffTSInput(
        data.plant_id, data.dataset, data.view,
        scale * data.xyz @ rotation.T, data.rgb, data.normals @ rotation.T,
        data.rgb_available,
        None if data.target_xyz is None else scale * data.target_xyz @ rotation.T,
        None if data.target_flow is None else data.target_flow @ rotation.T,
        data.target_parent,
    )
