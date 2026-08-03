#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tomato_recon.data.fruit_pseudo import write_empty_pseudo_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the optional pseudo-fruit cache contract (no labels are fabricated by default)"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="disabled_no_detector")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    args = parser.parse_args()
    path = write_empty_pseudo_manifest(
        args.output, method=args.method, threshold=args.min_confidence
    )
    print(path)


if __name__ == "__main__":
    main()

