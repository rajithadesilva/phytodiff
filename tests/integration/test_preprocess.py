from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from tests.fixtures import create_tomatowur_fixture
from tomato_recon.data.schemas import IGNORE_INDEX
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
                    "annotation_version": "0-paper-2Dto3D",
                    "split": "train",
                    "split_file": str(split),
                    "voxel_size_m": 0.001,
                    "skeleton_spacing_m": 0.02,
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
            self.assertTrue((processed / "samples/fixture_plant.graph.json").is_file())
            self.assertTrue((processed / "samples/fixture_plant.params.json").is_file())
            self.assertTrue((processed / "samples/fixture_plant.quality.json").is_file())
            graph = json.loads((processed / "samples/fixture_plant.graph.json").read_text())
            self.assertEqual(graph["coordinate_frame"], {"meters_per_unit": 1.0, "up_axis": "Z"})
            quality = json.loads(
                (processed / "samples/fixture_plant.quality.json").read_text()
            )
            self.assertIn(quality["status"], {"pass", "review"})
            self.assertEqual(quality["edge_count"], 4)


if __name__ == "__main__":
    unittest.main()
