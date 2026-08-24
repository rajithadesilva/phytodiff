#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tomato_recon.data.preprocess import preprocess_dataset


class ProgressPrinter:
    """Render compact live progress in a terminal and readable CI logs otherwise."""

    def __init__(self) -> None:
        self.stream = sys.stderr
        self.interactive = self.stream.isatty()
        self.line_width = 0

    def _write_instance_line(self, line: str, *, final: bool = False) -> None:
        if self.interactive:
            width = max(self.line_width, len(line))
            print(
                f"\r{line:<{width}}",
                end="\n" if final else "",
                file=self.stream,
                flush=True,
            )
            self.line_width = 0 if final else width
        else:
            print(line, file=self.stream, flush=True)

    def __call__(self, update: dict[str, Any]) -> None:
        phase = str(update["phase"])
        if phase == "loading_splits":
            splits = ", ".join(update["splits"])
            print(
                f"Loading {update['dataset']} splits ({splits}) from {update['raw_root']}",
                file=self.stream,
                flush=True,
            )
            print(f"Output dataset: {update['output_root']}", file=self.stream, flush=True)
            return
        if phase == "discovered":
            counts = ", ".join(
                f"{name}={count}" for name, count in update["split_counts"].items()
            )
            print(
                f"Discovered {update['total']} point-cloud instances ({counts})",
                file=self.stream,
                flush=True,
            )
            return
        if phase == "finished":
            print(
                "Preprocessing finished: "
                f"{update['new_instance_count']} new, "
                f"{update['resumed_instance_count']} resumed, "
                f"{update['skipped_instance_count']} unchanged; "
                f"next folder is plant_{update['next_plant_number']:06d}",
                file=self.stream,
                flush=True,
            )
            return

        prefix = (
            f"[{update['current']:>{len(str(update['total']))}}/{update['total']}] "
            f"{update['source_instance_id']} ({update['split']})"
        )
        target = update.get("instance_id")
        if phase == "checking":
            self._write_instance_line(f"{prefix}: checking inputs")
        elif phase == "processing":
            self._write_instance_line(f"{prefix} -> {target}: {update['action']}")
        elif phase == "writing":
            self._write_instance_line(f"{prefix} -> {target}: writing cache")
        elif phase == "skipped":
            elapsed = float(update["elapsed_seconds"])
            self._write_instance_line(
                f"{prefix} -> {target}: unchanged, skipped ({elapsed:.1f}s)", final=True
            )
        elif phase == "completed":
            elapsed = float(update["elapsed_seconds"])
            self._write_instance_line(
                f"{prefix} -> {target}: complete "
                f"({update['point_count']} points, {elapsed:.1f}s)",
                final=True,
            )


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
