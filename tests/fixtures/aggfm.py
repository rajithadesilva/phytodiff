"""Small CPU/GPU AG-GFM configurations and an isolated multi-source tree fixture."""

from pathlib import Path

from omegaconf import OmegaConf

from tomato_recon.config import load_config
from tomato_recon.data.processed import (
    make_tiny_sample, save_processed_sample, write_processed_dataset_manifest,
)
from tomato_recon.data.side import SideSettings, ensure_side
from tomato_recon.data.top_down import TopDownSettings, ensure_top_down


def tiny_config(*overrides):
    cfg, _ = load_config("aggfm", [
        "fruit.enabled=false", "data.max_nodes=16", "model.encoder.name=pointnext",
        "model.encoder.output_dim=16", "model.encoder.global_dim=24",
        "model.aggfm.hidden_dim=32", "model.aggfm.heads=4", "model.aggfm.blocks=2",
        "model.aggfm.observed_seeds=12", "model.aggfm.latent_seeds=4",
        "model.aggfm.memory_points=12", "model.aggfm.max_points=32",
        "model.aggfm.dropout=0.0", "inference.flow_steps=2", *overrides,
    ])
    return cfg


def write_dataset(root: Path):
    sources = ("tomatowur", "tomatopgt", "pheno4d")
    instances = []
    for source in sources:
        for split in ("train", "val", "test"):
            index = len(instances) + 1
            sample = make_tiny_sample(16, 64, seed=index)
            sample.plant_id = f"plant_{index:06d}"
            sample.metadata.update(instance_id=sample.plant_id, dataset=source,
                                   rgb_available=source != "pheno4d")
            entry = dict(dataset=source, plant_id=sample.plant_id, instance_id=sample.plant_id,
                         source_plant_id=f"{source}-{split}", source_instance_id=f"scan-{split}",
                         split=split, status="complete", preprocessing_hash="tiny-fixture-v1",
                         cache_file=f"{sample.plant_id}/full.npz")
            save_processed_sample(sample, root / entry["cache_file"])
            ensure_top_down(root, entry, TopDownSettings(occlusion_radius_m=0.02))
            ensure_side(root, entry, SideSettings(occlusion_radius_m=0.02))
            instances.append(entry)
    write_processed_dataset_manifest(root, dict(schema_version="1.0", layout="flat-plant-instance-v1",
        datasets={s: {"dataset": s, "preprocessing_hash": "tiny-fixture-v1"} for s in sources},
        instances=instances))
    return instances


def save_stage1(path, encoder, cfg, compatibility):
    import torch
    torch.save({"stage": "encoder", "model_state": encoder.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True), "max_nodes": 16,
                "pcl_types": list(cfg.data.pcl_types), "dataset_compatibility": compatibility}, path)
