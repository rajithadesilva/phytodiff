"""Pinned metadata and verification for external foundation checkpoints."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

SONATA_REPO_ID = "facebook/sonata"
SONATA_FILENAME = "sonata.pth"
SONATA_REVISION = "fe3c5c914663b83aa01643ea15a389a1e2ff2772"
SONATA_SHA256 = "c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50"
SONATA_SIZE = 434_008_287


@lru_cache(maxsize=4)
def _sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sonata_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Sonata checkpoint not found at {checkpoint}; run `make prepare-stage1-models`."
        )
    stat = checkpoint.stat()
    if stat.st_size != SONATA_SIZE:
        raise ValueError(
            f"Sonata checkpoint has size {stat.st_size}, expected {SONATA_SIZE}; "
            "run `make prepare-stage1-models` to replace it."
        )
    actual = _sha256(str(checkpoint.resolve()), stat.st_size, stat.st_mtime_ns)
    if actual != SONATA_SHA256:
        raise ValueError(
            f"Sonata checkpoint SHA256 mismatch: expected {SONATA_SHA256}, got {actual}."
        )
    return checkpoint
