"""Stage 1 feature extraction and DiffTS checkpoint contracts."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path

import torch

from tomato_recon.data.diffts import DiffTSInput
from tomato_recon.models.diffusion.diffts_model import DiffTSModel
from tomato_recon.models.encoders.base import PointBackbone
from tomato_recon.models.encoders.registry import create_backbone_from_config


UPSTREAM_COMMIT = "a1b75f5c4804ac018c553dcc6cf9a78bd518d8e4"
TRAIN_VIEWS = ("full", "top_down", "side")
POINTS_SCHEMA = "diffts-stage2-v1"
STAGE1_ADAPTER_SCHEMA = "diffts-stage2-v2"


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_stage2_schema(checkpoint: dict) -> None:
    variant = checkpoint.get("variant")
    schema = checkpoint.get("schema_version")
    if variant == "points" and schema == POINTS_SCHEMA:
        return
    if variant == "points_stage1" and schema == STAGE1_ADAPTER_SCHEMA:
        return
    if variant == "points_stage1" and schema == POINTS_SCHEMA:
        raise ValueError("old points_stage1 checkpoints cannot load with the Stage 1 adapter; start a new run")
    raise ValueError("old stage 2 checkpoints cannot be loaded as DiffTS")


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


def create_stage1_backbone(cfg, device: torch.device, *, load_source: bool) -> tuple[PointBackbone | None, str | None]:
    if str(cfg.model.diffusion.variant) == "points":
        return None, None
    backbone = create_backbone_from_config(cfg.model.encoder).to(device)
    if not load_source:
        return backbone, None
    path = Path(str(cfg.model.encoder.checkpoint))
    if not path.is_file():
        raise FileNotFoundError(f"all-view Stage 1 checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("pcl_types", ())) != TRAIN_VIEWS:
        raise ValueError("Stage 1 feature checkpoint must have all three training views")
    if int(checkpoint.get("max_nodes", -1)) != int(cfg.data.max_nodes):
        raise ValueError("Stage 1 feature checkpoint max_nodes mismatch")
    state = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint["model_state"].items()
        if key.startswith("backbone.")
    }
    backbone.load_state_dict(state)
    return backbone, file_hash(path)


def set_stage1_frozen(backbone: PointBackbone | None, frozen: bool) -> None:
    if backbone is None:
        return
    backbone.requires_grad_(not frozen)
    if frozen:
        backbone.eval()
    else:
        backbone.train()


def stage1_features(backbone: PointBackbone | None, data: DiffTSInput, *, frozen: bool = True):
    if backbone is None:
        return None, None
    mask = torch.ones((1, len(data.xyz)), dtype=torch.bool, device=data.xyz.device)
    context = torch.no_grad() if frozen else nullcontext()
    with context:
        points, _, global_feature = backbone(
            data.xyz[None], torch.cat((data.rgb, data.normals), -1)[None], mask
        )
    if points.shape[1] != len(data.xyz):
        raise RuntimeError("Stage 1 output points are not aligned with DiffTS inputs")
    return points[0], global_feature[0]


def save_stage2_checkpoint(path: Path, model, optimizer, cfg, epoch: int, metrics: dict,
                           stage1_hash: str | None, compatibility: dict, *,
                           backbone: PointBackbone | None = None, scaler=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model.variant == "points_stage1" and backbone is None:
        raise ValueError("points_stage1 checkpoints must include Stage 1 backbone weights")
    torch.save({
        "schema_version": POINTS_SCHEMA if model.variant == "points" else STAGE1_ADAPTER_SCHEMA,
        "stage": "diffts",
        "variant": model.variant,
        "upstream_commit": UPSTREAM_COMMIT,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": dict(cfg) if isinstance(cfg, dict) else __import__("omegaconf").OmegaConf.to_container(cfg, resolve=True),
        "stage1_checkpoint_sha256": stage1_hash,
        "stage1_frozen": bool(cfg.trainer.freeze_stage1) if backbone is not None else None,
        "stage1_backbone_state": backbone.state_dict() if backbone is not None else None,
        "dataset_compatibility": compatibility,
        "pcl_types": list(TRAIN_VIEWS),
        "max_nodes": model.max_nodes,
        "metrics": metrics,
    }, path)


def load_stage2_checkpoint(path: str | Path, model: DiffTSModel, *,
                           stage1_hash: str | None = None,
                           compatibility: dict | None = None,
                           optimizer=None, backbone: PointBackbone | None = None,
                           checkpoint: dict | None = None, scaler=None) -> dict:
    checkpoint = checkpoint if checkpoint is not None else torch.load(path, map_location="cpu", weights_only=False)
    require_stage2_schema(checkpoint)
    if checkpoint.get("variant") != model.variant:
        raise ValueError("DiffTS checkpoint conditioning variant mismatch")
    if stage1_hash is not None and checkpoint.get("stage1_checkpoint_sha256") != stage1_hash:
        raise ValueError("DiffTS checkpoint Stage 1 feature source mismatch")
    if model.variant == "points" and checkpoint.get("stage1_checkpoint_sha256") is not None:
        raise ValueError("DiffTS points checkpoint has an unexpected Stage 1 feature source")
    if checkpoint.get("max_nodes") != model.max_nodes:
        raise ValueError("DiffTS checkpoint max_nodes mismatch")
    if tuple(checkpoint.get("pcl_types", ())) != TRAIN_VIEWS:
        raise ValueError("DiffTS checkpoint training views mismatch")
    if compatibility is not None and checkpoint.get("dataset_compatibility", {}).get("signature") != compatibility.get("signature"):
        raise ValueError("DiffTS checkpoint dataset compatibility mismatch")
    if model.variant == "points_stage1":
        if not checkpoint.get("stage1_checkpoint_sha256"):
            raise ValueError("points_stage1 checkpoint is missing Stage 1 source provenance")
        if backbone is None or checkpoint.get("stage1_backbone_state") is None:
            raise ValueError("points_stage1 checkpoint requires embedded Stage 1 backbone weights")
        backbone.load_state_dict(checkpoint["stage1_backbone_state"])
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and scaler.is_enabled() and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint
