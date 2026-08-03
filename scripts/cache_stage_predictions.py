#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Version cached stage predictions")
    parser.add_argument("--stage", required=True, choices=["encoder", "diffusion", "graph", "parametric"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.input.is_file():
        raise FileNotFoundError("checkpoint and input prediction file must exist")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / args.input.name
    shutil.copy2(args.input, destination)
    manifest = {
        "schema_version": "1.0",
        "stage": args.stage,
        "checkpoint_sha256": sha256(args.checkpoint),
        "prediction_file": destination.name,
        "prediction_sha256": sha256(destination),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

