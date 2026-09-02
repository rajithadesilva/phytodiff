#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from tomato_recon.data.preprocess import preprocess_dataset
from tomato_recon.data.progress import ProgressPrinter


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic TomatoWUR v3 caches")
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if "data" in cfg:
        cfg = cfg.data
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    manifest = preprocess_dataset(cfg, progress=ProgressPrinter())
    print(
        json.dumps(
            {
                "manifest": str(Path(cfg.processed_root) / "manifest.json"),
                "sample_count": manifest["sample_count"],
                "instance_count": manifest["instance_count"],
                "plant_count": manifest["plant_count"],
                "source_plant_count": manifest["source_plant_count"],
                "point_cloud_count": manifest["point_cloud_count"],
                "new_instance_count": manifest["new_instance_count"],
                "resumed_instance_count": manifest["resumed_instance_count"],
                "skipped_instance_count": manifest["skipped_instance_count"],
                "next_plant_number": manifest["next_plant_number"],
                "split_counts": manifest["split_counts"],
                "skeleton_source": manifest["skeleton_source"],
                "skeleton_annotation_version": manifest["skeleton_annotation_version"],
                "skeleton_modified_count": manifest["skeleton_modified_count"],
                "warnings": manifest["warnings"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
