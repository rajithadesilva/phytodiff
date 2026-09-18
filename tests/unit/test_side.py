from __future__ import annotations

import numpy as np
import pytest

from tomato_recon.data.side import SideSettings, side_indices


def test_y_stacked_surfaces_and_isolated_points() -> None:
    xyz = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 0], [0, 1, 0]], dtype=float)
    np.testing.assert_array_equal(side_indices(xyz), [1, 2, 3])


def test_tolerance_zero_radius_and_invalid_occluders() -> None:
    xyz = np.array([[0, 0, 0], [0, 0.125, 0], [0, 4, 0]], dtype=float)
    valid = np.array([True, True, False])
    np.testing.assert_array_equal(
        side_indices(xyz, valid, depth_tolerance_m=0.125), [0, 1]
    )
    np.testing.assert_array_equal(
        side_indices(xyz, valid, depth_tolerance_m=0.124), [1]
    )
    np.testing.assert_array_equal(
        side_indices(xyz, valid, occlusion_radius_m=0), [0, 1]
    )


def test_increasing_radius_is_monotonic_deterministic_and_matches_oracle() -> None:
    rng = np.random.default_rng(12)
    xyz = rng.uniform(-0.01, 0.01, (600, 3))
    previous = set(range(len(xyz)))
    for radius in [0, 0.001, 0.002, 0.005, 0.01]:
        indices = side_indices(xyz, occlusion_radius_m=radius)
        assert len(indices) > 0
        assert np.all(np.diff(indices) > 0)
        assert set(indices) <= previous
        np.testing.assert_array_equal(
            indices, side_indices(xyz, occlusion_radius_m=radius)
        )
        expected = []
        for index, point in enumerate(xyz):
            nearby = np.linalg.norm(xyz[:, [0, 2]] - point[[0, 2]], axis=1) <= radius
            if radius == 0 or xyz[nearby, 1].max() - point[1] <= 0.001:
                expected.append(index)
        np.testing.assert_array_equal(indices, expected)
        previous = set(indices)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("parameter", ["occlusion_radius_m", "depth_tolerance_m"])
def test_rejects_invalid_settings(parameter: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        side_indices(np.zeros((1, 3)), **{parameter: value})


def test_rejects_invalid_coordinates_masks_and_enabled() -> None:
    with pytest.raises(ValueError, match="finite array"):
        side_indices(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite array"):
        side_indices(np.array([[0, np.nan, 0]]))
    with pytest.raises(ValueError, match="at least one valid"):
        side_indices(np.zeros((1, 3)), np.array([False]))
    with pytest.raises(ValueError, match="boolean array"):
        side_indices(np.zeros((1, 3)), np.ones(1, dtype=int))
    with pytest.raises(ValueError, match="boolean"):
        SideSettings(enabled="false")
