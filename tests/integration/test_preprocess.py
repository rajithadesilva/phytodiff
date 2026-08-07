from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from tests.fixtures import create_tomatowur_fixture
from tomato_recon.data.schemas import IGNORE_INDEX, OrganType
from tomato_recon.data.preprocess import preprocess_dataset
from tomato_recon.data.tomatowur import ProcessedTomatoDataset


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_official_csv_layout_to_versioned_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            split = create_tomatowur_fixture(root)
            processed = Path(directory) / "processed"
            cfg = OmegaConf.create(
                {
                    "raw_root": str(root),
                    "processed_root": str(processed),
                    "annotation_version": "0-paper-2Dto3D_improved",
                    "skeleton_mode": "official_gt_direct",
                    "split": "train",
                    "split_file": str(split),
                    "voxel_size_m": 0.001,
                    "max_nodes": 16,
                    "num_points": 100,
                    "use_rgb": True,
                    "use_normals": True,
                    "remove_support_pole": True,
                    "visibility_distance_m": 0.01,
                    "strict": True,
                }
            )
            first = preprocess_dataset(cfg)
            first_manifest = (processed / "manifest.json").read_text(encoding="utf-8")
            second = preprocess_dataset(cfg)
            self.assertEqual(first["preprocessing_hash"], second["preprocessing_hash"])
            self.assertEqual(first_manifest, (processed / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(first["sample_count"], 1)
            dataset = ProcessedTomatoDataset(processed)
            sample = dataset[0]
            sample.validate()
            self.assertEqual(sample.plant_id, "fixture_plant")
            self.assertFalse(bool((sample.semantic == 3).any()))
            self.assertTrue(bool((sample.semantic == IGNORE_INDEX).any()))
            self.assertEqual(sample.metadata["support_pole_point_count"], 5)
            self.assertEqual(sample.metadata["skeleton_source"], "official_ground_truth")
            self.assertFalse(sample.metadata["skeleton_modified"])
            self.assertEqual(int(sample.node_valid.sum()), 5)
            self.assertEqual(sample.parent_index[:5].tolist(), [-1, 0, 1, 1, 3])
            self.assertAlmostEqual(float(sample.node_xyz[2, 2]), 0.16)
            self.assertAlmostEqual(float(sample.node_xyz[3, 0]), 0.05)
            self.assertAlmostEqual(float(sample.node_xyz[3, 2]), 0.11)
            self.assertEqual(
                sample.organ_type[:5].tolist(),
                [
                    int(OrganType.MAIN_STEM),
                    int(OrganType.MAIN_STEM),
                    int(OrganType.MAIN_STEM),
                    int(OrganType.SIDE_STEM),
                    int(OrganType.SIDE_STEM),
                ],
            )
            self.assertEqual(sample.metadata["official_gt_node_ids"], [0, 1, 2, 3, 4])
            self.assertEqual(sample.metadata["official_gt_parent_ids"], [None, 0, 1, 1, 3])
            self.assertEqual(sample.metadata["official_gt_edge_types"], ["", "<", "<", "+", "<"])
            self.assertTrue((processed / "samples/fixture_plant.graph.json").is_file())
            self.assertTrue((processed / "samples/fixture_plant.params.json").is_file())
            self.assertFalse((processed / "samples/fixture_plant.quality.json").exists())
            graph = json.loads((processed / "samples/fixture_plant.graph.json").read_text())
            self.assertEqual(graph["coordinate_frame"], {"meters_per_unit": 1.0, "up_axis": "Z"})
            self.assertEqual(
                [edge["edge_type"] for edge in graph["edges"]],
                ["continuation", "continuation", "attachment", "continuation"],
            )
            self.assertEqual(first["skeleton_modified_count"], 0)
            self.assertEqual(first["skeleton_annotation_version"], "0-paper-2Dto3D_improved")

    def test_combines_official_splits_without_cross_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            train_path = create_tomatowur_fixture(root)
            train_entry = json.loads(train_path.read_text(encoding="utf-8"))[0]
            split_dir = train_path.parent
            for split, plant_id in (("val", "fixture_val"), ("test", "fixture_test")):
                entry = dict(train_entry)
                entry["plant_id"] = plant_id
                (split_dir / f"{split}.json").write_text(
                    json.dumps([entry]), encoding="utf-8"
                )

            processed = Path(directory) / "processed"
            cfg = OmegaConf.create(
                {
                    "raw_root": str(root),
                    "processed_root": str(processed),
                    "annotation_version": "0-paper-2Dto3D_improved",
                    "skeleton_mode": "official_gt_direct",
                    "split": "train",
                    "splits": ["train", "val", "test"],
                    "split_file": None,
                    "voxel_size_m": 0.001,
                    "max_nodes": 16,
                    "num_points": 100,
                    "use_rgb": True,
                    "use_normals": True,
                    "remove_support_pole": True,
                    "visibility_distance_m": 0.01,
                    "strict": True,
                }
            )
            manifest = preprocess_dataset(cfg)

            self.assertEqual(manifest["sample_count"], 3)
            self.assertEqual(manifest["split_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(len(ProcessedTomatoDataset(processed, split="train")), 1)
            self.assertEqual(len(ProcessedTomatoDataset(processed, split="val")), 1)
            test_dataset = ProcessedTomatoDataset(processed, split="test")
            self.assertEqual(len(test_dataset), 1)
            self.assertEqual(test_dataset[0].plant_id, "fixture_test")
            self.assertEqual(test_dataset[0].metadata["split"], "test")


if __name__ == "__main__":
    unittest.main()
