from __future__ import annotations

import numpy as np
import pytest

from tomato_recon.data.top_down import TopDownSettings, top_down_indices


def test_stacked_surfaces_and_isolated_points() -> None:
    xyz = np.array([[0, 0, 0], [0, 0, 1], [2, 0, 0], [0, 0, 1]], dtype=float)
    np.testing.assert_array_equal(top_down_indices(xyz), [1, 2, 3])


def test_tolerance_and_invalid_occluders() -> None:
    xyz = np.array([[0, 0, 0], [0, 0, 0.125], [0, 0, 4]], dtype=float)
    valid = np.array([True, True, False])
    np.testing.assert_array_equal(top_down_indices(xyz, valid, depth_tolerance_m=0.125), [0, 1])
    np.testing.assert_array_equal(top_down_indices(xyz, valid, depth_tolerance_m=0.124), [1])
    np.testing.assert_array_equal(top_down_indices(xyz, valid, occlusion_radius_m=0), [0, 1])


def test_increasing_radius_is_monotonic_and_deterministic() -> None:
    rng = np.random.default_rng(5)
    xyz = rng.uniform(-0.01, 0.01, (800, 3))
    previous = set(range(len(xyz)))
    for radius in [0, 0.001, 0.002, 0.005, 0.01]:
        indices = top_down_indices(xyz, occlusion_radius_m=radius)
        assert len(indices) > 0
        assert np.all(np.diff(indices) > 0)
        assert set(indices) <= previous
        np.testing.assert_array_equal(indices, top_down_indices(xyz, occlusion_radius_m=radius))
        # Independent brute-force oracle checks spatial batching and radius boundaries.
        expected = []
        for i, point in enumerate(xyz):
            nearby = np.linalg.norm(xyz[:, :2] - point[:2], axis=1) <= radius
            if radius == 0 or xyz[nearby, 2].max() - point[2] <= 0.001:
                expected.append(i)
        np.testing.assert_array_equal(indices, expected)
        previous = set(indices)


def test_dense_query_batches_and_exact_radius_boundary() -> None:
    xyz = np.zeros((1100, 3))
    xyz[:, 2] = np.arange(len(xyz))
    np.testing.assert_array_equal(top_down_indices(xyz), [1099])
    xyz = np.array([[0, 0, 0], [0.125, 0, 1], [0.2501, 0, 2]])
    np.testing.assert_array_equal(top_down_indices(xyz, occlusion_radius_m=0.125), [1, 2])


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("parameter", ["occlusion_radius_m", "depth_tolerance_m"])
def test_rejects_invalid_settings(parameter: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        top_down_indices(np.zeros((1, 3)), **{parameter: value})


@pytest.mark.parametrize("xyz", [np.zeros((2, 2)), np.array([[0, 0, np.nan]])])
def test_rejects_invalid_coordinates(xyz: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite array"):
        top_down_indices(xyz)


def test_rejects_empty_or_invalid_mask() -> None:
    for xyz, mask in [(np.empty((0, 3)), None), (np.zeros((1, 3)), np.array([False]))]:
        with pytest.raises(ValueError, match="at least one valid"):
            top_down_indices(xyz, mask)
    for mask in [np.ones(2, dtype=bool), np.ones(1, dtype=int)]:
        with pytest.raises(ValueError, match="boolean array"):
            top_down_indices(np.zeros((1, 3)), mask)
    with pytest.raises(ValueError, match="boolean"):
        TopDownSettings(enabled="false")
