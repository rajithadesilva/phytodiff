#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from tomato_recon.data.progress import ProgressPrinter
from tomato_recon.data.tomatopgt import preprocess_tomatopgt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert complete TomatoPGT scans into canonical dataset instances"
    )
    parser.add_argument("--config", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if "data" in cfg:
        cfg = cfg.data
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    report = preprocess_tomatopgt(cfg, progress=ProgressPrinter())
    print(
        json.dumps(
            {
                "manifest": str(Path(cfg.processed_root) / "manifest.json"),
                "instance_count": report["instance_count"],
                "source_plant_count": report["source_plant_count"],
                "new_instance_count": report["new_instance_count"],
                "resumed_instance_count": report["resumed_instance_count"],
                "skipped_instance_count": report["skipped_instance_count"],
                "ignored_incomplete_count": report["ignored_incomplete_count"],
                "next_plant_number": report["next_plant_number"],
                "split_counts": report["split_counts"],
                "skeleton_source": report["skeleton_source"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

