from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tomato_recon.data.preprocess import (
    apply_inverse_normalisation,
    canonical_hash,
    normalise_coordinates,
    pad_ground_truth_skeleton,
    voxel_downsample,
)
from tomato_recon.data.processed import (
    PointCloudViewDataset,
    ProcessedPlantDataset,
    make_tiny_sample,
    processed_dataset_compatibility,
    save_processed_sample,
    write_processed_dataset_manifest,
)
from tomato_recon.data.side import SideSettings, ensure_side
from tomato_recon.data.top_down import TopDownSettings, ensure_top_down


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

    def test_flat_dataset_can_select_one_source_or_the_combination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = {name: {"dataset": name, "preprocessing_hash": f"hash-{name}"} for name in ("tomatowur", "pheno4d")}
            instances = []
            for number, (dataset, split) in enumerate(
                (("tomatowur", "train"), ("pheno4d", "train"), ("pheno4d", "val")),
                start=1,
            ):
                instance_id = f"plant_{number:06d}"
                instances.append(
                    {
                        "instance_id": instance_id,
                        "plant_number": number,
                        "plant_id": instance_id,
                        "source_plant_id": f"source-{number}",
                        "source_instance_id": f"scan-{number}",
                        "dataset": dataset,
                        "split": split,
                        "status": "complete",
                        "cache_file": f"{instance_id}/sample.npz",
                        "preprocessing_hash": f"hash-{dataset}",
                    }
                )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "layout": "flat-plant-instance-v1",
                        "datasets": datasets,
                        "instances": instances,
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                len(ProcessedPlantDataset(root, split="train", dataset="combined")), 2
            )
            pheno4d = ProcessedPlantDataset(root, split="train", dataset="pheno4d")
            self.assertEqual(len(pheno4d), 1)
            self.assertEqual(pheno4d.dataset_ids, ("pheno4d",))
            combined = processed_dataset_compatibility(root, "combined")
            self.assertEqual(set(combined["datasets"]), {"tomatowur", "pheno4d"})
            self.assertTrue(combined["signature"])
            self.assertNotEqual(
                combined["signature"],
                processed_dataset_compatibility(root, "pheno4d")["signature"],
            )
            with self.assertRaisesRegex(ValueError, "not available"):
                ProcessedPlantDataset(root, dataset="tomatopgt")

    def test_point_cloud_view_dataset_preserves_configured_view_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = make_tiny_sample(max_nodes=16, num_points=96)
            sample.plant_id = "plant_000001"
            sample.metadata["instance_id"] = sample.plant_id
            source_path = root / sample.plant_id / "sample.npz"
            save_processed_sample(sample, source_path)
            entry = {
                "dataset": "fixture",
                "plant_id": sample.plant_id,
                "instance_id": sample.plant_id,
                "source_plant_id": "source-1",
                "source_instance_id": "scan-1",
                "split": "train",
                "status": "complete",
                "cache_file": "plant_000001/sample.npz",
            }
            write_processed_dataset_manifest(
                root,
                {
                    "schema_version": "1.0",
                    "layout": "flat-plant-instance-v1",
                    "datasets": {"fixture": {"dataset": "fixture"}},
                    "instances": [entry],
                },
            )
            ensure_top_down(root, entry, TopDownSettings(occlusion_radius_m=0.02))
            ensure_side(root, entry, SideSettings(occlusion_radius_m=0.02))

            full = PointCloudViewDataset(
                root, split="train", dataset="fixture", pcl_types=["full"]
            )
            top_down = PointCloudViewDataset(
                root, split="train", dataset="fixture", pcl_types=["top_down"]
            )
            views = PointCloudViewDataset(
                root,
                split="train",
                dataset="fixture",
                pcl_types=["side", "full", "top_down"],
            )

            self.assertEqual((len(full), len(top_down), len(views)), (1, 1, 3))
            self.assertEqual(full[0].metadata["pcl_type"], "full")
            self.assertEqual(top_down[0].metadata["pcl_type"], "top_down")
            self.assertEqual(
                [views[index].metadata["pcl_type"] for index in range(3)],
                ["side", "full", "top_down"],
            )
            self.assertLess(len(top_down[0].xyz), len(full[0].xyz))
            for field in (
                "node_xyz", "node_valid", "parent_flow", "parent_index",
                "organ_type", "topology_role", "visibility",
            ):
                torch.testing.assert_close(getattr(top_down[0], field), getattr(full[0], field))
            for invalid in ([], ["full", "full"], ["partial"], "full"):
                with self.assertRaisesRegex(ValueError, "data.pcl_types"):
                    PointCloudViewDataset(root, pcl_types=invalid)

if __name__ == "__main__":
    unittest.main()
