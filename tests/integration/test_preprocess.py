from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from tests.fixtures import create_tomatowur_fixture
from tomato_recon.data.processed import ProcessedPlantDataset
from tomato_recon.data.schemas import IGNORE_INDEX, OrganType
from tomato_recon.data.preprocess import preprocess_dataset
from tomato_recon.data.top_down import file_sha256


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_rejects_output_outside_dataset_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = OmegaConf.create(
                {
                    "dataset_id": "tomatowur",
                    "dataset_name": "TomatoWUR",
                    "dataset_version": "3",
                    "dataset_root": str(Path(directory) / "dataset"),
                    "processed_root": str(Path(directory) / "processed"),
                }
            )

            with self.assertRaisesRegex(ValueError, "must equal data.dataset_root"):
                preprocess_dataset(cfg)

    def test_official_csv_layout_to_versioned_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            split = create_tomatowur_fixture(root)
            dataset_root = Path(directory) / "dataset"
            dataset_root.mkdir(parents=True)
            (dataset_root / "plant_000007").mkdir()
            # An on-disk folder can be newer than the last manifest entry (for
            # example, after an interrupted or manually recovered conversion).
            (dataset_root / "plant_000011").mkdir()
            (dataset_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "layout": "flat-plant-instance-v1",
                        "datasets": {
                            "pheno4d": {
                                "dataset": "pheno4d",
                                "dataset_name": "Pheno4D",
                                "dataset_version": "1",
                            }
                        },
                        "instances": [
                            {
                                "dataset": "pheno4d",
                                "instance_id": "plant_000007",
                                "source_instance_id": "T01_0305_a",
                                "plant_id": "plant_000007",
                                "source_plant_id": "Tomato01",
                                "cache_file": "plant_000007/full.npz",
                                "split": "train",
                                "status": "processing",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cfg = OmegaConf.create(
                {
                    "dataset_id": "tomatowur",
                    "dataset_name": "TomatoWUR",
                    "dataset_version": "3",
                    "raw_root": str(root),
                    "dataset_root": str(dataset_root),
                    "processed_root": str(dataset_root),
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
            first_manifest = (dataset_root / "manifest.json").read_text(encoding="utf-8")
            progress_updates: list[dict] = []
            second = preprocess_dataset(cfg, progress=progress_updates.append)
            self.assertEqual(first["preprocessing_hash"], second["preprocessing_hash"])
            self.assertEqual(
                first_manifest, (dataset_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first["sample_count"], 1)
            self.assertEqual(first["instance_count"], 1)
            self.assertEqual(first["plant_count"], 1)
            self.assertEqual(first["source_plant_count"], 1)
            self.assertEqual(first["dataset"], "tomatowur")
            self.assertEqual(first["dataset_name"], "TomatoWUR")
            self.assertEqual(first["dataset_version"], "3")
            self.assertEqual(first["layout"], "flat-plant-instance-v1")
            self.assertEqual(first["new_instance_count"], 1)
            self.assertEqual(first["next_plant_number"], 13)
            self.assertEqual(second["new_instance_count"], 0)
            self.assertEqual(second["skipped_instance_count"], 1)
            partial_path = dataset_root / "plant_000012/top_down.npz"
            side_path = partial_path.with_name("side.npz")
            self.assertTrue(partial_path.is_file())
            self.assertTrue(side_path.is_file())
            full_path = partial_path.with_name("full.npz")
            full_hash = file_sha256(full_path)
            partial_path.unlink()
            side_path.unlink()
            backfill = preprocess_dataset(cfg)
            self.assertEqual(backfill["skipped_instance_count"], 1)
            self.assertTrue(partial_path.is_file())
            self.assertTrue(side_path.is_file())
            cfg.top_down = {"occlusion_radius_m": 0.0}
            cfg.side = {"occlusion_radius_m": 0.0}
            changed = preprocess_dataset(cfg)
            self.assertEqual(changed["skipped_instance_count"], 1)
            self.assertEqual(changed["preprocessing_hash"], first["preprocessing_hash"])
            self.assertEqual(file_sha256(full_path), full_hash)
            self.assertEqual(
                changed["instances"][0]["top_down"]["point_count"],
                changed["instances"][0]["point_count"],
            )
            self.assertEqual(
                changed["instances"][0]["side"]["point_count"],
                changed["instances"][0]["point_count"],
            )
            self.assertEqual(progress_updates[0]["phase"], "loading_splits")
            self.assertEqual(progress_updates[-1]["phase"], "finished")
            self.assertEqual(
                [update["phase"] for update in progress_updates].count("skipped"), 1
            )
            collection = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(collection["layout"], "flat-plant-instance-v1")
            self.assertIn("pheno4d", collection["datasets"])
            self.assertIn("tomatowur", collection["datasets"])
            self.assertEqual(collection["next_plant_number"], 13)
            self.assertEqual(collection["plant_count"], 1)
            self.assertEqual(collection["source_plant_count"], 1)
            dataset = ProcessedPlantDataset(dataset_root)
            sample = dataset[0]
            sample.validate()
            self.assertEqual(sample.plant_id, "plant_000012")
            self.assertEqual(sample.metadata["instance_id"], "plant_000012")
            self.assertEqual(sample.metadata["plant_id"], "plant_000012")
            self.assertEqual(sample.metadata["source_instance_id"], "fixture_plant")
            self.assertEqual(sample.metadata["source_plant_id"], "fixture_plant")
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
            instance_dir = dataset_root / "plant_000012"
            self.assertTrue((instance_dir / "full.npz").is_file())
            self.assertTrue((instance_dir / "full.graph.json").is_file())
            self.assertTrue((instance_dir / "full.params.json").is_file())
            self.assertFalse((instance_dir / "quality.json").exists())
            self.assertFalse((dataset_root / "tomatowur").exists())
            graph = json.loads((instance_dir / "full.graph.json").read_text())
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
                point_cloud = root / "point_clouds" / f"{plant_id}.csv"
                shutil.copyfile(train_entry["file_name"], point_cloud)
                entry["file_name"] = str(point_cloud)
                entry["instance_id"] = plant_id
                entry["plant_id"] = plant_id
                (split_dir / f"{split}.json").write_text(
                    json.dumps([entry]), encoding="utf-8"
                )

            dataset_root = Path(directory) / "dataset"
            cfg = OmegaConf.create(
                {
                    "dataset_id": "tomatowur",
                    "dataset_name": "TomatoWUR",
                    "dataset_version": "3",
                    "raw_root": str(root),
                    "dataset_root": str(dataset_root),
                    "processed_root": str(dataset_root),
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
            self.assertEqual(len(ProcessedPlantDataset(dataset_root, split="train")), 1)
            self.assertEqual(len(ProcessedPlantDataset(dataset_root, split="val")), 1)
            test_dataset = ProcessedPlantDataset(dataset_root, split="test")
            self.assertEqual(len(test_dataset), 1)
            self.assertEqual(test_dataset[0].plant_id, "plant_000003")
            self.assertEqual(test_dataset[0].metadata["source_plant_id"], "fixture_test")
            self.assertEqual(test_dataset[0].metadata["split"], "test")

    def test_each_point_cloud_is_a_separate_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            split_path = create_tomatowur_fixture(root)
            entries = json.loads(split_path.read_text(encoding="utf-8"))
            second = dict(entries[0])
            second_point_cloud = root / "point_clouds" / "fixture_plant_scan_2.csv"
            shutil.copyfile(entries[0]["file_name"], second_point_cloud)
            second["file_name"] = str(second_point_cloud)
            entries.append(second)
            split_path.write_text(json.dumps(entries), encoding="utf-8")

            dataset_root = Path(directory) / "dataset"
            cfg = OmegaConf.create(
                {
                    "dataset_id": "tomatowur",
                    "dataset_name": "TomatoWUR",
                    "dataset_version": "3",
                    "raw_root": str(root),
                    "dataset_root": str(dataset_root),
                    "processed_root": str(dataset_root),
                    "annotation_version": "0-paper-2Dto3D_improved",
                    "skeleton_mode": "official_gt_direct",
                    "split": "train",
                    "split_file": str(split_path),
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

            self.assertEqual(manifest["instance_count"], 2)
            self.assertEqual(manifest["point_cloud_count"], 2)
            self.assertEqual(manifest["plant_count"], 2)
            self.assertEqual(manifest["source_plant_count"], 1)
            self.assertEqual(
                [entry["instance_id"] for entry in manifest["instances"]],
                ["plant_000001", "plant_000002"],
            )
            self.assertEqual(
                [entry["source_instance_id"] for entry in manifest["instances"]],
                ["fixture_plant", "fixture_plant_scan_2"],
            )
            self.assertEqual(len(ProcessedPlantDataset(dataset_root)), 2)
            self.assertTrue((dataset_root / "plant_000001/full.npz").is_file())
            self.assertTrue((dataset_root / "plant_000002/full.npz").is_file())


if __name__ == "__main__":
    unittest.main()
