from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.visualize_dataset import (
    _camera_basis,
    _perspective,
    load_side,
    load_top_down,
    render,
)
from tomato_recon.data.processed import make_tiny_sample, save_processed_sample
from tomato_recon.data.side import SideSettings, ensure_side
from tomato_recon.data.top_down import TopDownSettings, ensure_top_down


def test_side_camera_projects_height_up_and_near_points_larger() -> None:
    basis = _camera_basis(35, 15)
    torch.testing.assert_close(basis @ basis.T, torch.eye(3))
    points = torch.stack([torch.zeros(3), torch.tensor([0., 0., 1.]),
                          basis[0], basis[0] + basis[2]])
    projected, depth = _perspective(points, torch.zeros(3), basis, 5.)
    assert projected[1, 1] > projected[0, 1]
    assert projected[3, 0] > projected[2, 0]
    assert depth[3] < depth[2]


def test_nine_panel_render_uses_saved_visibility_and_handles_missing(tmp_path: Path) -> None:
    sample = make_tiny_sample()
    path = tmp_path / "sample.npz"
    save_processed_sample(sample, path)
    assert load_top_down(path, sample) is None
    render(sample, tmp_path / "missing.png")
    entry = {"cache_file": "sample.npz"}
    ensure_top_down(tmp_path, entry, TopDownSettings(occlusion_radius_m=0.02))
    ensure_side(tmp_path, entry, SideSettings(occlusion_radius_m=0.02))
    top_down = load_top_down(path, sample)
    side = load_side(path, sample)
    assert 0 < len(top_down.source_indices) < len(sample.xyz)
    assert 0 < len(side.source_indices) <= len(sample.xyz)
    render(sample, tmp_path / "preview.png", top_down, side)
    with Image.open(tmp_path / "preview.png") as image:
        assert image.size == (1560, 1612)
        overlay = np.asarray(image)[572:, 1040:]
        assert np.any(np.all(overlay == [20, 140, 70], axis=-1))
        assert np.any(np.all(overlay == [180, 187, 197], axis=-1))
    with Image.open(tmp_path / "missing.png") as image:
        assert image.size == (1560, 1612)


def test_stale_cloud_is_rejected(tmp_path: Path) -> None:
    sample = make_tiny_sample()
    path = tmp_path / "sample.npz"
    save_processed_sample(sample, path)
    ensure_top_down(tmp_path, {"cache_file": "sample.npz"}, TopDownSettings())
    sample.xyz[0, 0] += 1
    save_processed_sample(sample, path)
    with pytest.raises(ValueError, match="stale"):
        load_top_down(path, sample)
