#!/usr/bin/env python3
"""Download and verify pinned external Stage 1 model artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tomato_recon.models.pretrained import (
    SONATA_FILENAME,
    SONATA_REPO_ID,
    SONATA_REVISION,
    SONATA_SHA256,
    SONATA_SIZE,
    verify_sonata_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pretrained/sonata"))
    parser.add_argument("--check", action="store_true", help="Only verify the local artifact")
    args = parser.parse_args()
    checkpoint = args.output / SONATA_FILENAME

    if not args.check:
        try:
            verify_sonata_checkpoint(checkpoint)
            print(f"Sonata checkpoint already verified: {checkpoint}", flush=True)
        except (FileNotFoundError, ValueError):
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "huggingface-hub is required; rebuild the Docker training image."
                ) from exc
            args.output.mkdir(parents=True, exist_ok=True)
            print(
                f"Downloading {SONATA_REPO_ID}/{SONATA_FILENAME} at {SONATA_REVISION} "
                f"({SONATA_SIZE / 1_000_000:.1f} MB)...",
                flush=True,
            )
            downloaded = Path(
                hf_hub_download(
                    repo_id=SONATA_REPO_ID,
                    filename=SONATA_FILENAME,
                    revision=SONATA_REVISION,
                    local_dir=args.output,
                    force_download=True,
                )
            )
            if downloaded.resolve() != checkpoint.resolve():
                raise RuntimeError(
                    f"Hugging Face returned unexpected checkpoint path {downloaded}"
                )

    verify_sonata_checkpoint(checkpoint)
    manifest = {
        "schema_version": "1.0",
        "repo_id": SONATA_REPO_ID,
        "filename": SONATA_FILENAME,
        "revision": SONATA_REVISION,
        "sha256": SONATA_SHA256,
        "size": SONATA_SIZE,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Verified Sonata checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
