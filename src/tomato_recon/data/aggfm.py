"""AG-GFM adapter: full skeleton supervision; strictly point-only standalone input."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tomato_recon.data.diffts import from_sample as diffts_from_sample, rgb_available
from tomato_recon.data.schemas import PlantSample


def dataset_compatibility(root, selection):
    """Preprocessing provenance plus exact plant/split membership for this backend."""
    from tomato_recon.data.processed import (
        load_processed_dataset_manifest, processed_dataset_compatibility,
    )
    compatibility = processed_dataset_compatibility(root, selection)
    manifest = load_processed_dataset_manifest(root)
    instances = sorted((x["dataset"], x["source_plant_id"], x["instance_id"], x.get("split"))
                       for x in manifest["instances"] if x.get("status") == "complete"
                       and (selection == "combined" or x["dataset"] == selection))
    compatibility["split_instances"] = instances
    compatibility["split_signature"] = hashlib.sha256(json.dumps(instances).encode()).hexdigest()
    return compatibility


@dataclass
class AGGFMInput:
    plant_id: str
    dataset: str
    view: str
    xyz: Tensor
    rgb: Tensor
    normals: Tensor
    rgb_available: bool
    metadata: dict
    target_xyz: Tensor | None = None
    target_parent: Tensor | None = None
    organ_type: Tensor | None = None
    topology_role: Tensor | None = None
    visibility: Tensor | None = None

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                               for f in fields(self) if isinstance(getattr(self, f.name), Tensor)})

    def points_only(self):
        return replace(self, target_xyz=None, target_parent=None, organ_type=None,
                       topology_role=None, visibility=None)


def from_sample(sample: PlantSample) -> AGGFMInput:
    base = diffts_from_sample(sample)
    valid = sample.node_valid.bool()
    return AGGFMInput(
        base.plant_id, base.dataset, base.view, base.xyz, base.rgb, base.normals,
        base.rgb_available, sample.metadata, base.target_xyz, base.target_parent,
        sample.organ_type[valid], sample.topology_role[valid], sample.visibility[valid],
    )


def sample_points(data: AGGFMInput, limit: int, seed: int) -> AGGFMInput:
    if limit < 2:
        raise ValueError("max_points must be at least two")
    if len(data.xyz) <= limit:
        return data
    indices = torch.randperm(len(data.xyz), generator=torch.Generator().manual_seed(seed))[:limit]
    return replace(data, xyz=data.xyz[indices], rgb=data.rgb[indices], normals=data.normals[indices])


def load_input(path: str | Path, view: str = "full") -> AGGFMInput:
    """Read only point arrays and metadata, even if a full cache also contains GT.

    Coordinates must already satisfy the preprocessing contract (metres, root centred,
    Z up). The saved inverse transform is retained for downstream export.
    """
    if view not in {"full", "top_down", "side"}:
        raise ValueError(f"invalid input view {view!r}")
    with np.load(path, allow_pickle=False) as cloud:
        metadata = json.loads(str(cloud["metadata_json"].item()))
        if view != "full":
            from tomato_recon.data.side import SIDE_SPEC
            from tomato_recon.data.top_down import TOP_DOWN_SPEC

            spec = TOP_DOWN_SPEC if view == "top_down" else SIDE_SPEC
            if metadata.get(view, {}).get("algorithm_version") != spec.algorithm_version:
                raise ValueError(f"stale or invalid {view} point view")
        valid = cloud["point_valid"].astype(bool)
        xyz, rgb, normals = [torch.from_numpy(cloud[key][valid].copy()).float()
                             for key in ("xyz", "rgb", "normals")]
        plant_id = str(cloud["plant_id"].item())
    if len(xyz) < 2 or any(not torch.isfinite(v).all() for v in (xyz, rgb, normals)):
        raise ValueError("input needs at least two finite points with finite features")
    available = rgb_available(metadata)
    return AGGFMInput(plant_id, str(metadata["dataset"]), view, xyz,
                      rgb if available else torch.zeros_like(rgb), normals, available, metadata)
