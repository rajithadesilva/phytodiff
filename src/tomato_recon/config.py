"""Small Hydra-compatible configuration loader used by all entry points."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


STAGE_CONFIGS = {
    "encoder": "encoder/kpconvx.yaml",
    "pointnext": "encoder/pointnext.yaml",
    "sonata_ptv3": "encoder/sonata_ptv3.yaml",
    "kpconvx": "encoder/kpconvx.yaml",
    "diffusion": "diffusion/default.yaml",
    "graph": "graph/default.yaml",
    "parametric": "parametric/default.yaml",
    "joint": "experiment/combined_stage1_K256.yaml",
    "infer": "infer.yaml",
    "smoke": "smoke/all.yaml",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_cli(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-name")
    parser.add_argument("--config")
    parser.add_argument("--resume")
    parser.add_argument("--help", action="store_true")
    return parser.parse_known_args(argv)


def load_config(
    stage: str,
    argv: Sequence[str] | None = None,
    *,
    config_path: str | Path | None = None,
) -> tuple[DictConfig, argparse.Namespace]:
    known, overrides = _parse_cli(argv)
    root = repository_root()
    base = OmegaConf.load(root / "configs/config.yaml")
    data = OmegaConf.load(root / "configs/data/combined.yaml")
    cfg = OmegaConf.merge(base, {"data": data})
    stage_relative = STAGE_CONFIGS.get(stage, f"{stage}/default.yaml")
    stage_path = root / "configs" / stage_relative
    if stage_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(stage_path))
    requested = known.config or config_path
    if requested:
        selected_path = Path(requested)
        if not selected_path.is_absolute():
            selected_path = root / selected_path
    else:
        name = known.config_name or stage
        relative = STAGE_CONFIGS.get(name, STAGE_CONFIGS.get(stage, f"{stage}/default.yaml"))
        selected_path = root / "configs" / relative
    if selected_path.exists() and selected_path.resolve() != stage_path.resolve():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(selected_path))
    elif not selected_path.exists():
        raise FileNotFoundError(f"configuration not found: {selected_path}")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    validate_config(cfg, stage)
    return cfg, known


def validate_config(cfg: DictConfig, stage: str | None = None) -> None:
    from tomato_recon.data.processed import (
        normalise_dataset_selection,
        normalise_point_cloud_type,
    )

    cfg.data.dataset = normalise_dataset_selection(cfg.data.get("dataset", "combined"))
    cfg.data.pcl_type = normalise_point_cloud_type(cfg.data.get("pcl_type", "full"))
    if int(cfg.data.max_nodes) <= 1:
        raise ValueError("data.max_nodes must be greater than one")
    diffusion_k = int(cfg.model.diffusion.max_nodes)
    if diffusion_k != int(cfg.data.max_nodes):
        raise ValueError("data.max_nodes and model.diffusion.max_nodes must match")
    if str(cfg.export.up_axis) != "Z" or float(cfg.export.meters_per_unit) != 1.0:
        raise ValueError("USD export currently supports only Z-up and meters_per_unit=1.0")
    if bool(cfg.fruit.enabled) and not Path(str(cfg.fruit.pseudo_labels_dir)).is_dir():
        warnings.warn(
            "fruit pseudo-label cache is absent; disabling the optional fruit branch/losses",
            RuntimeWarning,
            stacklevel=2,
        )
        cfg.fruit.enabled = False
    if stage in {"encoder", "joint", "infer", None}:
        from tomato_recon.models.encoders.registry import ensure_backbone_available

        ensure_backbone_available(str(cfg.model.encoder.name))


def config_hash(cfg: DictConfig) -> str:
    payload = OmegaConf.to_container(cfg, resolve=True)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def save_resolved_config(cfg: DictConfig, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, path)
    return path
