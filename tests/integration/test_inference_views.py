from __future__ import annotations

import json
from pathlib import Path

import pytest

from tomato_recon.data.processed import make_tiny_sample, save_processed_sample
from tomato_recon.data.side import SideSettings, ensure_side
from tomato_recon.data.top_down import TopDownSettings, ensure_top_down
from tomato_recon.infer import main


def test_processed_multiview_inference_fans_out_outputs(tmp_path: Path) -> None:
    sample = make_tiny_sample(max_nodes=16, num_points=64)
    sample.plant_id = "plant_000001"
    sample.metadata["instance_id"] = sample.plant_id
    source = tmp_path / sample.plant_id / "full.npz"
    save_processed_sample(sample, source)
    entry = {"cache_file": f"{sample.plant_id}/full.npz"}
    ensure_top_down(tmp_path, entry, TopDownSettings(occlusion_radius_m=0.02))
    ensure_side(tmp_path, entry, SideSettings(occlusion_radius_m=0.02))
    output = tmp_path / "inference"
    main(
        [
            "--config",
            "configs/smoke/all.yaml",
            f"input.path={source}",
            "data.pcl_types=[side,full,top_down]",
            f"output.dir={output}",
            f"model.pipeline_checkpoint={tmp_path / 'missing.ckpt'}",
            "export.usd=false",
        ]
    )
    manifest = json.loads((output / "inference_manifest.json").read_text())
    assert manifest["pcl_types"] == ["side", "full", "top_down"]
    assert list(manifest["views"]) == ["full", "side", "top_down"]  # sorted JSON keys
    for view in manifest["pcl_types"]:
        assert (output / view / "plant_graph.json").is_file()
        assert (output / view / "input_normalised.ply").is_file()


def test_raw_inference_rejects_derived_view_selection(tmp_path: Path) -> None:
    source = tmp_path / "scan.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"requires data.pcl_types=\[full\]"):
        main(
            [
                "--config",
                "configs/smoke/all.yaml",
                f"input.path={source}",
                "data.pcl_types=[side]",
                f"output.dir={tmp_path / 'output'}",
            ]
        )
