from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.generate_side import main
from tests.integration.test_top_down import create_dataset
from tomato_recon.data.processed import processed_dataset_compatibility
from tomato_recon.data.side import (
    ALGORITHM_VERSION,
    SideSettings,
    ensure_side,
    generate_side_dataset,
)
from tomato_recon.data.top_down import POINT_FIELDS, file_sha256


def test_side_backfill_alignment_metadata_repair_and_full_contract(tmp_path: Path) -> None:
    original = create_dataset(tmp_path)
    hashes = {path: file_sha256(path) for path in tmp_path.glob("plant_*/sample.*")}
    compatibility = processed_dataset_compatibility(tmp_path)
    report = generate_side_dataset(tmp_path)
    assert report["generated"] == 2 and not report["failures"]
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for entry, old_entry in zip(manifest["instances"], original["instances"], strict=True):
        assert {key: value for key, value in entry.items() if key != "side"} == old_entry
        with np.load(tmp_path / entry["side"]["cache_file"]) as partial:
            with np.load(tmp_path / entry["cache_file"]) as full:
                indices = partial["source_point_indices"]
                np.testing.assert_array_equal(indices, [0, 1, 2])
                for key in POINT_FIELDS:
                    np.testing.assert_array_equal(partial[key], full[key][indices])
            metadata = json.loads(partial["metadata_json"].item())
            assert metadata["point_to_original_index"] == [10, 20, 30]
            assert metadata["side"]["view_direction"] == [0.0, -1.0, 0.0]
            assert metadata["side"]["sensor_side"] == "+Y"
            assert metadata["side"]["projection_axes"] == ["X", "Z"]
            assert metadata["side"]["algorithm_version"] == ALGORITHM_VERSION
            assert metadata["side"]["graph_target_file"] == "sample.graph.json"
            assert entry["side"]["retained_fraction"] == 1.0
    assert generate_side_dataset(tmp_path)["skipped"] == 2
    assert (tmp_path / "manifest.json").read_bytes() == manifest_bytes
    for damage in ("missing", "corrupt"):
        path = tmp_path / "plant_000001/side.npz"
        path.unlink() if damage == "missing" else path.write_bytes(b"corrupt")
        repaired = generate_side_dataset(tmp_path)
        assert repaired["generated"] == 1 and repaired["skipped"] == 1
    assert processed_dataset_compatibility(tmp_path) == compatibility
    assert all(file_sha256(path) == digest for path, digest in hashes.items())


def test_side_changed_source_settings_and_version_regenerate(tmp_path: Path) -> None:
    entry = create_dataset(tmp_path)["instances"][0]
    settings = SideSettings()
    assert ensure_side(tmp_path, entry, settings) == "generated"
    assert ensure_side(tmp_path, entry, settings) == "skipped"
    entry["side"]["algorithm_version"] = "old"
    assert ensure_side(tmp_path, entry, settings) == "generated"
    assert ensure_side(tmp_path, entry, SideSettings(occlusion_radius_m=0)) == "generated"
    assert ensure_side(tmp_path, entry, SideSettings(enabled=False)) == "disabled"


def test_side_failures_continue_cli_and_atomic_write(tmp_path: Path) -> None:
    manifest = create_dataset(tmp_path)
    entry = manifest["instances"][0]
    ensure_side(tmp_path, entry, SideSettings())
    previous = dict(entry["side"])
    path = tmp_path / previous["cache_file"]
    before = path.read_bytes()
    with patch(
        "tomato_recon.data.top_down.np.savez_compressed",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            ensure_side(tmp_path, entry, SideSettings(occlusion_radius_m=0))
    assert path.read_bytes() == before
    assert entry["side"] == previous

    (tmp_path / "plant_000001/sample.npz").write_bytes(b"broken source")
    assert main([str(tmp_path)]) == 1
    assert (tmp_path / "plant_000002/side.npz").is_file()
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), "--occlusion-radius-m", "nan"])
    assert error.value.code == 2
