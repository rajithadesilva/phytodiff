"""Complete Stage 1 extraction, provenance, compatible and resumable checkpoints."""

from contextlib import nullcontext
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config

from .flow import integrate
from .model import AGGFMModel
from .topology import decode

SCHEMA = "aggfm-stage2-v1"
ARCHITECTURE_VERSION = 1
ENCODER_KEYS = ("name", "input_dim", "output_dim", "global_dim", "num_semantic_classes",
                "grid_size", "shell_sizes", "layer_blocks", "init_channels", "channel_scaling",
                "radius_scaling", "neighbor_limits", "kernel_radius", "attention_groups")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_model(cfg):
    return AGGFMModel(int(cfg.model.encoder.output_dim), int(cfg.model.encoder.global_dim), cfg.model.aggfm)


def check_preprocessing(compatibility, available):
    for source, hashes in compatibility["datasets"].items():
        if not set(hashes).issubset(available.get("datasets", {}).get(source, [])):
            raise ValueError(f"Stage 1 preprocessing is incompatible with {source}")


def check_evaluation_compatibility(compatibility, available):
    check_preprocessing(compatibility, available)
    if "split_instances" in available:
        sources = set(compatibility["datasets"])
        saved = sorted(tuple(x) for x in available["split_instances"] if x[0] in sources)
        if saved != sorted(tuple(x) for x in compatibility["split_instances"]):
            raise ValueError("AG-GFM evaluation split assignment mismatch")


def create_stage1(cfg, device, *, compatibility=None, source=True):
    encoder = PointEncoder(create_backbone_from_config(cfg.model.encoder),
                           int(cfg.model.encoder.num_semantic_classes)).to(device)
    if not source:
        return encoder, None
    path = Path(str(cfg.model.encoder.checkpoint))
    if not path.is_file():
        raise FileNotFoundError(f"complete Stage 1 checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != "encoder":
        raise ValueError("AG-GFM requires a complete Stage 1 encoder checkpoint")
    saved = payload.get("config", {}).get("model", {}).get("encoder", {})
    current = OmegaConf.to_container(cfg.model.encoder, resolve=True)
    required = ENCODER_KEYS if current["name"] == "kpconvx" else ENCODER_KEYS[:5]
    if any(key not in saved for key in required):
        raise ValueError("Stage 1 checkpoint is missing architecture metadata")
    for key in ENCODER_KEYS:
        if key in saved and saved[key] != current.get(key):
            raise ValueError(f"Stage 1 architecture mismatch: {key}")
    if compatibility is not None:
        check_preprocessing(compatibility, payload.get("dataset_compatibility", {}))
    # Strict loading includes semantic, skeleton, offset and junction heads.
    encoder.load_state_dict(payload["model_state"], strict=True)
    return encoder, file_hash(path)


def extract_stage1(encoder, data, *, frozen=True):
    if frozen:
        encoder.eval()
    context = torch.no_grad() if frozen else nullcontext()
    with context:
        rgb = data.rgb if data.rgb_available else torch.zeros_like(data.rgb)
        output = encoder(data.xyz[None], torch.cat((rgb, data.normals), -1)[None],
                         torch.ones((1, len(data.xyz)), dtype=torch.bool, device=data.xyz.device))
    if output.point_features.shape[1] != len(data.xyz):
        raise RuntimeError("Stage 1 features must align with input points")
    return output


@torch.no_grad()
def predict(model, encoder, data, cfg, *, seed=None, steps=None, on_step=None):
    """Only point fields enter conditioning; target fields are never consulted."""
    model.eval()
    encoder.eval()
    generator = torch.Generator(device=data.xyz.device).manual_seed(int(cfg.seed if seed is None else seed))
    output = extract_stage1(encoder, data.points_only())
    seeds, memory = model.condition(output, generator=generator)
    state, heads = integrate(model, seeds, memory, steps=int(cfg.inference.flow_steps if steps is None else steps),
                              solver=str(cfg.inference.solver), on_step=on_step)
    prediction = decode(state, heads, seeds, data.plant_id, cfg.inference)
    prediction.graph.source.update(dataset=data.dataset, view=data.view,
                                   normalised_to_original=data.metadata.get("normalised_to_original"))
    return prediction, seeds


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def require_schema(payload):
    if payload.get("schema_version") != SCHEMA or payload.get("stage") != "aggfm":
        raise ValueError("checkpoint is not AG-GFM v1")
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("AG-GFM architecture version mismatch")


def save_checkpoint(path, model, encoder, optimizer, cfg, epoch, metrics, stage1_hash,
                    compatibility, *, scheduler=None, scaler=None, stale=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA, "architecture_version": ARCHITECTURE_VERSION, "stage": "aggfm",
        "epoch": epoch, "model_state": model.state_dict(), "stage1_state": encoder.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": OmegaConf.to_container(cfg, resolve=True), "metrics": metrics, "stale": stale,
        "stage1_checkpoint_sha256": stage1_hash, "stage1_checkpoint": str(cfg.model.encoder.checkpoint),
        "dataset_compatibility": compatibility, "dataset": str(cfg.data.dataset),
        "pcl_types": list(cfg.data.pcl_types), "max_nodes": int(cfg.data.max_nodes),
        "random_seed": int(cfg.seed), "rng_state": rng_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, encoder, cfg, *, compatibility=None, stage1_hash=None,
                    optimizer=None, scheduler=None, scaler=None, resume=False, payload=None):
    payload = payload if payload is not None else torch.load(path, map_location="cpu", weights_only=False)
    require_schema(payload)
    saved = OmegaConf.create(payload["config"])
    if payload["dataset"] != str(cfg.data.dataset) or payload["pcl_types"] != list(cfg.data.pcl_types):
        raise ValueError("AG-GFM dataset/view selection mismatch")
    if payload["max_nodes"] != int(cfg.data.max_nodes):
        raise ValueError("AG-GFM max_nodes mismatch")
    for section in ("model.aggfm", "model.encoder"):
        before, after = [OmegaConf.to_container(OmegaConf.select(c, section), resolve=True)
                         for c in (saved, cfg)]
        if section == "model.encoder":
            before, after = [{key: val.get(key) for key in ENCODER_KEYS} for val in (before, after)]
        if before != after:
            raise ValueError(f"AG-GFM configuration mismatch: {section}")
    if resume:
        for key in ("freeze_stage1", "batch_size", "learning_rate", "stage1_learning_rate", "validation_view"):
            if saved.trainer.get(key) != cfg.trainer.get(key):
                raise ValueError(f"AG-GFM resume trainer mismatch: {key}")
        if int(saved.seed) != int(cfg.seed):
            raise ValueError("AG-GFM resume seed mismatch")
        for key in ("flow_steps", "solver", "existence_threshold", "min_nodes", "max_edge_length_m"):
            if saved.inference.get(key) != cfg.inference.get(key):
                raise ValueError(f"AG-GFM resume validation inference mismatch: {key}")
    if compatibility is not None and payload["dataset_compatibility"]["signature"] != compatibility["signature"]:
        raise ValueError("AG-GFM dataset compatibility mismatch")
    if not payload.get("stage1_checkpoint_sha256"):
        raise ValueError("AG-GFM checkpoint missing Stage 1 provenance")
    if stage1_hash is not None and payload["stage1_checkpoint_sha256"] != stage1_hash:
        raise ValueError("AG-GFM Stage 1 source mismatch")
    model.load_state_dict(payload["model_state"])
    encoder.load_state_dict(payload["stage1_state"])
    for component, key in ((optimizer, "optimizer_state"), (scheduler, "scheduler_state"), (scaler, "scaler_state")):
        if component is not None and payload.get(key) is not None:
            component.load_state_dict(payload[key])
    if resume:
        restore_rng(payload["rng_state"])
    return payload


def load_runtime(path, *, device=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_schema(payload)
    cfg = OmegaConf.create(payload["config"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg).to(device)
    encoder, _ = create_stage1(cfg, device, source=False)
    load_checkpoint(path, model, encoder, cfg, payload=payload)
    return model.eval(), encoder.eval(), cfg, payload, device
