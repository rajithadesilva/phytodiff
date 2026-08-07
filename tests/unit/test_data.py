from __future__ import annotations

import unittest

import numpy as np

from tomato_recon.data.preprocess import (
    apply_inverse_normalisation,
    canonical_hash,
    normalise_coordinates,
    pad_ground_truth_skeleton,
    voxel_downsample,
)


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.xyz = np.asarray(
            [[0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 0, 1], [2, 0, 1]],
            dtype=np.float32,
        )
        self.parent = np.asarray([-1, 0, 1, 1, 3], dtype=np.int64)
        self.edge_type = np.asarray(["", "<", "<", "+", "<"], dtype=object)

    def test_coordinate_round_trip(self) -> None:
        points = self.xyz + np.array([1.0, 2.0, 3.0])
        normalised, nodes, transform = normalise_coordinates(points, points, 0)
        np.testing.assert_allclose(apply_inverse_normalisation(normalised, transform), points)
        np.testing.assert_allclose(nodes[0], 0)

    def test_fixed_k_padding(self) -> None:
        xyz, parent, edge_type, valid = pad_ground_truth_skeleton(
            self.xyz, self.parent, self.edge_type, 8
        )
        self.assertEqual(xyz.shape, (8, 3))
        self.assertEqual(parent.shape, (8,))
        self.assertEqual(valid.sum(), 5)
        np.testing.assert_array_equal(xyz[:5], self.xyz)
        np.testing.assert_array_equal(parent[:5], self.parent)
        np.testing.assert_array_equal(edge_type[:5], self.edge_type)
        self.assertTrue(np.all(parent[5:] == -1))

    def test_ground_truth_is_never_reduced(self) -> None:
        with self.assertRaisesRegex(ValueError, "increase max_nodes"):
            pad_ground_truth_skeleton(self.xyz, self.parent, self.edge_type, 4)

    def test_voxel_mapping_is_repeatable(self) -> None:
        values = np.asarray([[0, 0, 0], [0.0002, 0, 0], [0.002, 0, 0]], dtype=np.float32)
        first = voxel_downsample(values, 0.001, np.arange(3))
        second = voxel_downsample(values, 0.001, np.arange(3))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[2], second[2])

    def test_canonical_cache_hash(self) -> None:
        self.assertEqual(canonical_hash({"b": 2, "a": 1}), canonical_hash({"a": 1, "b": 2}))

if __name__ == "__main__":
    unittest.main()
