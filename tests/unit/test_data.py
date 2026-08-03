from __future__ import annotations

import unittest

import numpy as np

from tomato_recon.data.preprocess import (
    apply_inverse_normalisation,
    canonical_hash,
    fixed_k_skeleton,
    normalise_coordinates,
    topology_preserving_reduce,
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
        xyz, parent, _, valid, reduced = fixed_k_skeleton(
            self.xyz, self.parent, self.edge_type, 8
        )
        self.assertEqual(xyz.shape, (8, 3))
        self.assertEqual(parent.shape, (8,))
        self.assertEqual(valid.sum(), 5)
        self.assertFalse(reduced)
        self.assertTrue(np.all(parent[5:] == -1))

    def test_topology_preserving_reduction(self) -> None:
        dense = np.asarray([[0, 0, float(z)] for z in range(9)] + [[1, 0, 4], [2, 0, 4]], dtype=np.float32)
        parent = np.asarray([-1, 0, 1, 2, 3, 4, 5, 6, 7, 4, 9])
        edge = np.asarray([""] + ["<"] * 8 + ["+", "<"], dtype=object)
        xyz, reduced_parent, _, kept = topology_preserving_reduce(dense, parent, edge, 7)
        self.assertEqual(len(xyz), 7)
        self.assertIn(0, kept)  # root
        self.assertIn(4, kept)  # junction
        self.assertIn(8, kept)  # main tip
        self.assertIn(10, kept)  # lateral tip
        self.assertEqual((reduced_parent < 0).sum(), 1)
        self.assertTrue(np.all(reduced_parent[1:] < np.arange(1, len(reduced_parent))))

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

