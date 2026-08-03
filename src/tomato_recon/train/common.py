from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from tomato_recon.config import save_resolved_config
from tomato_recon.data.collate import collate_plant_samples
from tomato_recon.data.schemas import PlantBatch, PlantSample
from tomato_recon.data.tomatowur import ProcessedTomatoDataset, make_tiny_sample


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(cfg: DictConfig) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and int(cfg.trainer.devices) > 0 else "cpu")


def load_training_sample(cfg: DictConfig) -> PlantSample:
    manifest = Path(cfg.data.processed_root) / "manifest.json"
    if manifest.is_file():
        dataset = ProcessedTomatoDataset(cfg.data.processed_root, split=str(cfg.data.split))
        if not len(dataset):
            raise ValueError(f"processed split {cfg.data.split!r} contains no samples")
        return dataset[0]
    if bool(cfg.trainer.fast_dev_run):
        return make_tiny_sample(int(cfg.data.max_nodes), min(int(cfg.data.num_points), 96), int(cfg.seed))
    raise FileNotFoundError(
        f"processed dataset not found at {cfg.data.processed_root}; run Stage 0 preprocessing first"
    )


def load_training_batch(cfg: DictConfig, device: torch.device) -> PlantBatch:
    return collate_plant_samples([load_training_sample(cfg)]).to(device)


def create_training_loader(cfg: DictConfig) -> DataLoader | list[PlantBatch]:
    manifest = Path(cfg.data.processed_root) / "manifest.json"
    if manifest.is_file():
        dataset = ProcessedTomatoDataset(cfg.data.processed_root, split=str(cfg.data.split))
        generator = torch.Generator().manual_seed(int(cfg.seed))
        return DataLoader(
            dataset,
            batch_size=int(cfg.trainer.batch_size),
            shuffle=True,
            num_workers=int(cfg.trainer.num_workers),
            collate_fn=collate_plant_samples,
            generator=generator,
            persistent_workers=int(cfg.trainer.num_workers) > 0,
        )
    if bool(cfg.trainer.fast_dev_run):
        return [collate_plant_samples([load_training_sample(cfg)])]
    raise FileNotFoundError(
        f"processed dataset not found at {cfg.data.processed_root}; run Stage 0 preprocessing first"
    )


def epoch_range(cfg: DictConfig, start_epoch: int) -> range:
    if bool(cfg.trainer.fast_dev_run):
        return range(start_epoch, start_epoch + 1)
    return range(start_epoch, max(int(cfg.trainer.max_epochs), start_epoch + 1))


def stage_output_dir(cfg: DictConfig, stage: str) -> Path:
    output = Path(cfg.output.dir)
    if bool(cfg.trainer.fast_dev_run) and output.name != stage:
        output = output / stage
    output.mkdir(parents=True, exist_ok=True)
    return output


def git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, check=True, text=True, capture_output=True
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "unknown"

    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short"), "diff": run("diff")}


def environment_info() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "numpy", "scipy", "omegaconf", "hydra-core", "networkx", "usd-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": os.sys.version,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": packages,
    }


def write_run_metadata(cfg: DictConfig, output_dir: Path) -> None:
    save_resolved_config(cfg, output_dir)
    (output_dir / "environment.json").write_text(
        json.dumps(environment_info(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "git_state.json").write_text(
        json.dumps(git_state(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    *,
    stage: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    sample: PlantSample,
    epoch: int,
    metrics: dict[str, float],
    upstream: dict[str, str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = git_state()
    payload = {
        "schema_version": "1.0",
        "stage": stage,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "label_map": {0: "background", 1: "leaf", 2: "main_stem", 3: "support_pole", 4: "side_stem"},
        "preprocessing_hash": sample.metadata.get("preprocessing_hash"),
        "max_nodes": int(cfg.data.max_nodes),
        "git_commit": state["commit"],
        "git_dirty": bool(state["status"] and state["status"] != "unknown"),
        "upstream_checkpoint_hashes": upstream or {},
        "metrics": metrics,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    expected_preprocessing_hash: str | None = None,
    expected_max_nodes: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actual_hash = checkpoint.get("preprocessing_hash")
    if expected_preprocessing_hash and actual_hash != expected_preprocessing_hash:
        raise ValueError(
            f"checkpoint preprocessing hash mismatch: expected {expected_preprocessing_hash}, got {actual_hash}"
        )
    if expected_max_nodes is not None and int(checkpoint.get("max_nodes", -1)) != expected_max_nodes:
        raise ValueError(
            f"checkpoint max_nodes mismatch: expected {expected_max_nodes}, got {checkpoint.get('max_nodes')}"
        )
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def maybe_load_upstream(
    path: str | None,
    module: torch.nn.Module,
    sample: PlantSample,
    cfg: DictConfig,
) -> str | None:
    if path and Path(path).is_file():
        load_checkpoint(
            path,
            module,
            expected_preprocessing_hash=sample.metadata.get("preprocessing_hash"),
            expected_max_nodes=int(cfg.data.max_nodes),
        )
        return checkpoint_sha256(path)
    if not bool(cfg.trainer.fast_dev_run):
        raise FileNotFoundError(f"required upstream checkpoint not found: {path}")
    return None


def write_metrics(output_dir: Path, values: dict[str, float]) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
