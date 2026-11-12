#!/usr/bin/env python3
"""Add a fixed top-down cloud to every completed processed instance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tomato_recon.data.top_down import TopDownSettings, generate_top_down_dataset


def print_progress(update: dict[str, Any]) -> None:
    detail = update.get("error", f"{update.get('point_count', 0)} points")
    print(
        f"[{update['current']}/{update['total']}] {update['instance_id']}: "
        f"{update['action']} ({detail})", file=sys.stderr, flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("--occlusion-radius-m", type=float, default=0.001)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.001)
    args = parser.parse_args(argv)
    try:
        settings = TopDownSettings(
            occlusion_radius_m=args.occlusion_radius_m,
            depth_tolerance_m=args.depth_tolerance_m,
        )
        report = generate_top_down_dataset(args.processed_root, settings, progress=print_progress)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
