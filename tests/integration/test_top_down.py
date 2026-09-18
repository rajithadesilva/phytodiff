from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.generate_top_down import main
from tomato_recon.data.processed import processed_dataset_compatibility
from tomato_recon.data.top_down import (
    POINT_FIELDS,
    TopDownSettings,
    ensure_top_down,
    file_sha256,
    generate_top_down_dataset,
)


def create_dataset(root: Path) -> dict:
    entries = []
    for number in [1, 2]:
        plant_id = f"plant_{number:06d}"
        folder = root / plant_id
        folder.mkdir()
        xyz = np.array([[0, 0, 0], [0, 0, 1], [2, 0, 0], [0, 0, 9]], dtype=np.float32)
        metadata = {"point_to_original_index": [10, 20, 30, 40],
                    "normalised_to_original": np.eye(4).tolist(), "dataset": "fixture"}
        np.savez_compressed(
            folder / "full.npz", xyz=xyz, rgb=xyz + 1, normals=xyz + 2,
            semantic=np.arange(4), instance=np.arange(4) + 5,
            point_valid=np.array([True, True, True, False]),
            plant_id=np.asarray(plant_id), instance_id=np.asarray(plant_id),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        for suffix in ["graph", "params"]:
            (folder / f"sample.{suffix}.json").write_text("{}")
        entries.append({
            "dataset": "fixture", "instance_id": plant_id, "plant_id": plant_id,
            "source_plant_id": plant_id, "source_instance_id": plant_id,
            "cache_file": f"{plant_id}/full.npz", "status": "complete", "split": "train",
            "preprocessing_hash": "unchanged-full-contract",
            "cache_sha256": file_sha256(folder / "full.npz"),
        })
    manifest = {"schema_version": "1.0", "layout": "flat-plant-instance-v1",
                "datasets": {"fixture": {"dataset": "fixture"}}, "instances": entries}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_backfill_alignment_repair_and_full_contract(tmp_path: Path) -> None:
    original = create_dataset(tmp_path)
    hashes = {p: file_sha256(p) for p in tmp_path.glob("plant_*/full.*")}
    compatibility = processed_dataset_compatibility(tmp_path)
    report = generate_top_down_dataset(tmp_path)
    assert report["generated"] == 2 and not report["failures"]
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for entry, old_entry in zip(manifest["instances"], original["instances"]):
        assert {k: v for k, v in entry.items() if k != "top_down"} == old_entry
        assert (tmp_path / entry["top_down"]["cache_file"]).stat().st_mode & 0o777 == (
            (tmp_path / entry["cache_file"]).stat().st_mode & 0o666
        )
        with np.load(tmp_path / entry["top_down"]["cache_file"]) as partial:
            with np.load(tmp_path / entry["cache_file"]) as full:
                indices = partial["source_point_indices"]
                np.testing.assert_array_equal(indices, [1, 2])
                for key in POINT_FIELDS:
                    np.testing.assert_array_equal(partial[key], full[key][indices])
            metadata = json.loads(partial["metadata_json"].item())
            assert metadata["point_to_original_index"] == [20, 30]
            assert metadata["normalised_to_original"] == np.eye(4).tolist()
            assert metadata["top_down"]["full_point_cloud_file"] == "full.npz"
            assert metadata["top_down"]["graph_target_file"] == "full.graph.json"
            assert entry["top_down"]["retained_fraction"] == 2 / 3
    assert generate_top_down_dataset(tmp_path)["skipped"] == 2
    assert (tmp_path / "manifest.json").read_bytes() == manifest_bytes
    for damage in ["missing", "corrupt"]:
        path = tmp_path / "plant_000001/top_down.npz"
        if damage == "missing":
            path.unlink()
        else:
            path.write_bytes(b"corrupt")
        report = generate_top_down_dataset(tmp_path)
        assert report["generated"] == 1 and report["skipped"] == 1
    assert generate_top_down_dataset(
        tmp_path, TopDownSettings(occlusion_radius_m=0)
    )["generated"] == 2
    assert processed_dataset_compatibility(tmp_path) == compatibility
    assert all(file_sha256(path) == digest for path, digest in hashes.items())


def test_changed_source_and_version_regenerate(tmp_path: Path) -> None:
    manifest = create_dataset(tmp_path)
    entry = manifest["instances"][0]
    settings = TopDownSettings()
    assert ensure_top_down(tmp_path, entry, settings) == "generated"
    source_path = tmp_path / entry["cache_file"]
    with np.load(source_path) as source:
        arrays = dict(source)
    arrays["xyz"][2, 2] = 5
    np.savez_compressed(source_path, **arrays)
    assert ensure_top_down(tmp_path, entry, settings) == "generated"
    entry["top_down"]["algorithm_version"] = "old"
    assert ensure_top_down(tmp_path, entry, settings) == "generated"
    assert ensure_top_down(tmp_path, entry, TopDownSettings(enabled=False)) == "disabled"


def test_failures_continue_and_cli_returns_nonzero(tmp_path: Path) -> None:
    create_dataset(tmp_path)
    (tmp_path / "plant_000001/full.npz").write_bytes(b"broken source")
    assert main([str(tmp_path)]) == 1
    assert (tmp_path / "plant_000002/top_down.npz").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "top_down" not in manifest["instances"][0]
    assert "top_down" in manifest["instances"][1]
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), "--occlusion-radius-m", "nan"])
    assert error.value.code == 2


def test_atomic_write_failure_preserves_previous_artifact(tmp_path: Path) -> None:
    entry = create_dataset(tmp_path)["instances"][0]
    ensure_top_down(tmp_path, entry, TopDownSettings())
    previous = dict(entry["top_down"])
    path = tmp_path / previous["cache_file"]
    before = path.read_bytes()
    with patch("tomato_recon.data.top_down.np.savez_compressed", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            ensure_top_down(tmp_path, entry, TopDownSettings(occlusion_radius_m=0))
    assert path.read_bytes() == before
    assert entry["top_down"] == previous
    assert sorted(p.name for p in path.parent.glob("*.npz")) == ["full.npz", "top_down.npz"]
