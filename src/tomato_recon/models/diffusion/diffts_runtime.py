"""Frozen Stage 1 extraction and strict DiffTS checkpoint contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from tomato_recon.data.diffts import DiffTSInput
from tomato_recon.models.diffusion.diffts_model import DiffTSModel
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config


UPSTREAM_COMMIT = "a1b75f5c4804ac018c553dcc6cf9a78bd518d8e4"
TRAIN_VIEWS = ("full", "top_down", "side")


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_model(cfg) -> DiffTSModel:
    settings = cfg.model.diffusion
    return DiffTSModel(
        str(settings.variant), int(cfg.data.max_nodes),
        condition_resolution=float(settings.condition_resolution),
        node_resolution=float(settings.node_resolution),
        stage1_point_dim=int(cfg.model.encoder.output_dim),
        stage1_global_dim=int(cfg.model.encoder.global_dim),
        width_divisor=int(settings.width_divisor),
    )


def load_frozen_stage1(cfg, device: torch.device) -> tuple[PointEncoder | None, str | None]:
    if str(cfg.model.diffusion.variant) == "points":
        return None, None
    path = Path(str(cfg.model.encoder.checkpoint))
    if not path.is_file():
        raise FileNotFoundError(f"all-view Stage 1 checkpoint not found: {path}")
    encoder = PointEncoder(
        create_backbone_from_config(cfg.model.encoder),
        int(cfg.model.encoder.num_semantic_classes),
    ).to(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("pcl_types", ())) != TRAIN_VIEWS:
        raise ValueError("Stage 1 feature checkpoint must have all three training views")
    if int(checkpoint.get("max_nodes", -1)) != int(cfg.data.max_nodes):
        raise ValueError("Stage 1 feature checkpoint max_nodes mismatch")
    encoder.load_state_dict(checkpoint["model_state"])
    encoder.requires_grad_(False).eval()
    return encoder, file_hash(path)


@torch.no_grad()
def stage1_features(encoder: PointEncoder | None, data: DiffTSInput):
    if encoder is None:
        return None, None
    mask = torch.ones((1, len(data.xyz)), dtype=torch.bool, device=data.xyz.device)
    output = encoder(
        data.xyz[None], torch.cat((data.rgb, data.normals), -1)[None], mask
    )
    if not torch.equal(output.point_xyz[0], data.xyz):
        raise RuntimeError("Stage 1 output points are not aligned with DiffTS inputs")
    return output.point_features[0].detach(), output.global_feature[0].detach()


def save_stage2_checkpoint(path: Path, model, optimizer, cfg, epoch: int, metrics: dict,
                           stage1_hash: str | None, compatibility: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "diffts-stage2-v1",
        "stage": "diffts",
        "variant": model.variant,
        "upstream_commit": UPSTREAM_COMMIT,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": dict(cfg) if isinstance(cfg, dict) else __import__("omegaconf").OmegaConf.to_container(cfg, resolve=True),
        "stage1_checkpoint_sha256": stage1_hash,
        "dataset_compatibility": compatibility,
        "pcl_types": list(TRAIN_VIEWS),
        "max_nodes": model.max_nodes,
        "metrics": metrics,
    }, path)


def load_stage2_checkpoint(path: str | Path, model: DiffTSModel, *,
                           stage1_hash: str | None, compatibility: dict | None = None,
                           optimizer=None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "diffts-stage2-v1":
        raise ValueError("old stage 2 checkpoints cannot be loaded as DiffTS")
    if checkpoint.get("variant") != model.variant:
        raise ValueError("DiffTS checkpoint conditioning variant mismatch")
    if checkpoint.get("stage1_checkpoint_sha256") != stage1_hash:
        raise ValueError("DiffTS checkpoint Stage 1 feature source mismatch")
    if checkpoint.get("max_nodes") != model.max_nodes:
        raise ValueError("DiffTS checkpoint max_nodes mismatch")
    if tuple(checkpoint.get("pcl_types", ())) != TRAIN_VIEWS:
        raise ValueError("DiffTS checkpoint training views mismatch")
    if compatibility is not None and checkpoint.get("dataset_compatibility", {}).get("signature") != compatibility.get("signature"):
        raise ValueError("DiffTS checkpoint dataset compatibility mismatch")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
