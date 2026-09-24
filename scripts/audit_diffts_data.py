#!/usr/bin/env python3
"""Audit the prepared DiffTS splits, views, RGB provenance, and target counts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


VIEWS = ("full", "top_down", "side")


def audit(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    source_split = Counter()
    source_colour = Counter()
    view_split = Counter()
    node_counts = []
    for item in manifest["instances"]:
        if item.get("status") != "complete":
            raise ValueError(f"incomplete processed plant: {item['plant_id']}")
        source, split, plant = item["dataset"], item["split"], item["plant_id"]
        source_split[(source, split)] += 1
        for view in VIEWS:
            path = root / plant / f"{view}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            view_split[(split, view)] += 1
            if view != "full":
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata_json"].item()))
                    if metadata.get(view, {}).get("source_cache_sha256") != item["cache_sha256"]:
                        raise ValueError(f"{plant}: {view} is not aligned to the full-plant target")
                    if {"node_xyz", "node_valid", "parent_flow", "parent_index"} & set(data.files):
                        raise ValueError(f"{plant}: {view} unexpectedly stores skeleton targets")
                    if view == "top_down":
                        available = metadata.get("rgb_available", source != "pheno4d")
                        source_colour[(source, bool(available))] += 1
        with np.load(root / plant / "full.npz", allow_pickle=False) as data:
            node_counts.append(int(data["node_valid"].sum()))
    if manifest["split_counts"] != {"train": 92, "val": 29, "test": 42}:
        raise ValueError("DiffTS requires the preserved 92/29/42 plant split")
    if len(node_counts) != 163 or min(node_counts) < 2 or max(node_counts) > 256:
        raise ValueError("DiffTS expects 163 plants with 2–256 skeleton nodes")
    return {
        "schema_version": "diffts-data-audit-v1",
        "plant_count": len(node_counts),
        "split_counts": manifest["split_counts"],
        "source_split_counts": {
            f"{source}/{split}": count
            for (source, split), count in sorted(source_split.items())
        },
        "view_split_counts": {
            f"{split}/{view}": count
            for (split, view), count in sorted(view_split.items())
        },
        "source_rgb_availability": {
            f"{source}/{str(available).lower()}": count
            for (source, available), count in sorted(source_colour.items())
        },
        "target_node_count_range": [min(node_counts), max(node_counts)],
        "partial_views_use_existing_full_skeleton": True,
        "partial_views_contain_skeleton_targets": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/diffts/data_audit.json"))
    args = parser.parse_args()
    report = audit(args.processed_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
