"""Optional confidence-weighted pseudo-fruit cache contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PSEUDO_SCHEMA_VERSION = "1.0"


def load_fruit_pseudo_labels(path: str | Path, min_confidence: float = 0.7) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        return {
            "points": np.empty((0, 3), dtype=np.float32),
            "confidence": np.empty((0,), dtype=np.float32),
        }
    with np.load(path, allow_pickle=False) as cache:
        points = cache["points"].astype(np.float32)
        confidence = cache["confidence"].astype(np.float32)
    keep = confidence >= min_confidence
    return {"points": points[keep], "confidence": confidence[keep]}


def write_empty_pseudo_manifest(output_dir: str | Path, *, method: str, threshold: float) -> Path:
    """Write an explicit no-proposals result; this function never fabricates labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": PSEUDO_SCHEMA_VERSION,
                "method": method,
                "confidence_threshold": threshold,
                "supervision": "pseudo_optional",
                "include_in_primary_metrics": False,
                "samples": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path

