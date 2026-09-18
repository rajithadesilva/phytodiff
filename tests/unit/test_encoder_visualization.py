from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

from scripts.visualize_encoder_predictions import encoder_metrics, main, render_encoder_prediction
from scripts.visualize_encoder_predictions import _select_samples
from tomato_recon.config import load_config
from tomato_recon.data.collate import collate_plant_samples
from tomato_recon.data.processed import (
    ProcessedPlantDataset, make_tiny_sample, save_processed_sample,
    write_processed_dataset_manifest,
)
from tomato_recon.data.side import SideSettings, ensure_side
from tomato_recon.data.top_down import TopDownSettings, ensure_top_down, file_sha256
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.pointnext import PointNeXtAdapter
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.train.common import save_checkpoint


class EncoderVisualizationTests(unittest.TestCase):
    def _dataset(self, root: Path):
        sample = make_tiny_sample(max_nodes=16, num_points=96)
        sample.plant_id = "plant_000001"
        sample.metadata["instance_id"] = sample.plant_id
        path = root / sample.plant_id / "sample.npz"
        save_processed_sample(sample, path)
        entry = {
            "dataset": "fixture", "plant_id": sample.plant_id,
            "instance_id": sample.plant_id, "source_plant_id": "source_plant",
            "source_instance_id": "source_instance", "split": "test", "status": "complete",
            "cache_file": "plant_000001/sample.npz",
            "preprocessing_hash": "fixture-preprocessing-hash",
        }
        write_processed_dataset_manifest(root, {
            "schema_version": "1.0", "layout": "flat-plant-instance-v1",
            "datasets": {"fixture": {"dataset": "fixture"}}, "instances": [entry],
        })
        return ProcessedPlantDataset(root, split="test"), sample, entry

    def test_pcl_selection_drives_inference_and_retains_full_targets(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, original, entry = self._dataset(root)
            source_hash = file_sha256(dataset.paths[0])
            full = _select_samples(dataset, 0, [], "full")[0]
            ensure_top_down(root, entry, TopDownSettings(occlusion_radius_m=0.02))
            partial = _select_samples(dataset, 0, [full.plant_id], "top_down")[0]
            ensure_side(root, entry, SideSettings(occlusion_radius_m=0.02))
            side = _select_samples(dataset, 0, [full.plant_id], "side")[0]
            self.assertEqual(full.metadata["pcl_type"], "full")
            self.assertEqual(partial.metadata["pcl_type"], "top_down")
            self.assertEqual(side.metadata["pcl_type"], "side")
            self.assertLess(len(partial.xyz), len(full.xyz))
            rows = partial.metadata["source_point_indices"]
            for field in ("xyz", "rgb", "normals", "semantic", "instance", "point_valid"):
                torch.testing.assert_close(getattr(partial, field), getattr(full, field)[rows])
            for field in ("node_xyz", "node_valid", "parent_flow", "parent_index",
                          "organ_type", "topology_role", "visibility"):
                torch.testing.assert_close(getattr(partial, field), getattr(full, field))
            self.assertEqual(file_sha256(dataset.paths[0]), source_hash)
            self.assertEqual(len(original.xyz), len(full.xyz))

            batch = collate_plant_samples([partial])
            model = PointEncoder(
                PointNeXtAdapter(input_dim=6, output_dim=16, global_dim=24),
                num_semantic_classes=5,
            ).eval()
            with torch.inference_mode():
                output = model(batch.xyz, torch.cat([batch.rgb, batch.normals], dim=-1),
                               batch.point_valid)
                metrics = encoder_metrics(partial, output, skeleton_threshold_m=0.01,
                                          junction_threshold_multiplier=2.0,
                                          probability_threshold=0.5)
            self.assertEqual(output.semantic_logits.shape[1], len(partial.xyz))
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
            render_encoder_prediction(partial, output, metrics, root / "partial.png")
            self.assertTrue((root / "partial.png").is_file())

    def test_top_down_requires_current_artifact_and_valid_selection(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, sample, entry = self._dataset(root)
            with self.assertRaisesRegex(FileNotFoundError, "make generate-top-down"):
                _select_samples(dataset, 0, [], "top_down")
            with self.assertRaisesRegex(ValueError, "pcl_types"):
                _select_samples(dataset, 0, [], "invalid")
            with self.assertRaisesRegex(ValueError, "plant IDs not found"):
                _select_samples(dataset, 0, ["plant_000002"], "full")
            ensure_top_down(root, entry, TopDownSettings())
            sample.xyz[0, 0] += 0.001
            save_processed_sample(sample, dataset.paths[0])
            with self.assertRaisesRegex(ValueError, "stale"):
                _select_samples(dataset, 0, [], "top_down")

    def test_metrics_and_six_panel_render(self) -> None:
        sample = make_tiny_sample(max_nodes=16, num_points=96)
        batch = collate_plant_samples([sample])
        model = PointEncoder(
            PointNeXtAdapter(input_dim=6, output_dim=16, global_dim=24),
            num_semantic_classes=5,
        ).eval()
        with torch.inference_mode():
            output = model(
                batch.xyz,
                torch.cat([batch.rgb, batch.normals], dim=-1),
                batch.point_valid,
            )
            metrics = encoder_metrics(
                sample,
                output,
                skeleton_threshold_m=0.01,
                junction_threshold_multiplier=2.0,
                probability_threshold=0.5,
            )

        self.assertTrue(
            {
                "semantic_miou",
                "skeleton_precision",
                "skeleton_recall",
                "skeleton_f1",
                "centreline_offset_mae_m",
                "centreline_offset_score",
                "junction_precision",
                "junction_recall",
                "junction_f1",
                "overall_score",
            }.issubset(metrics)
        )
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "preview.png"
            render_encoder_prediction(
                sample, output, metrics, output_path, max_render_points=64
            )
            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as image:
                self.assertEqual(image.size, (6 * 390, 70 + 390 + 70))

    def test_multiview_command_fans_out_outputs_in_array_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, entry = self._dataset(root)
            ensure_top_down(root, entry, TopDownSettings(occlusion_radius_m=0.02))
            ensure_side(root, entry, SideSettings(occlusion_radius_m=0.02))
            cfg, _ = load_config(
                "encoder",
                [
                    "--config",
                    "configs/smoke/all.yaml",
                    f"data.processed_root={root}",
                    "data.dataset=fixture",
                    "data.pcl_types=[full]",
                ],
            )
            model = PointEncoder(
                create_backbone_from_config(cfg.model.encoder),
                int(cfg.model.encoder.num_semantic_classes),
            )
            checkpoint = root / "encoder.ckpt"
            save_checkpoint(
                checkpoint,
                stage="encoder",
                model=model,
                optimizer=torch.optim.Adam(model.parameters()),
                cfg=cfg,
                sample=dataset[0],
                epoch=0,
                metrics={},
            )
            output = root / "visualizations"
            argv = [
                "visualize_encoder_predictions.py",
                "--checkpoint",
                str(checkpoint),
                "--processed-root",
                str(root),
                "--split",
                "test",
                "--dataset",
                "fixture",
                "--count",
                "1",
                "--pcl-types",
                "side",
                "full",
                "top_down",
                "--output",
                str(output),
                "--device",
                "cpu",
                "--max-render-points",
                "64",
            ]
            with patch("sys.argv", argv):
                main()
            root_metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(root_metrics["pcl_types"], ["side", "full", "top_down"])
            self.assertEqual(root_metrics["view_sample_counts"], {
                "side": 1, "full": 1, "top_down": 1,
            })
            for view in root_metrics["pcl_types"]:
                self.assertTrue((output / view / "plant_000001.png").is_file())
                report = json.loads((output / view / "metrics.json").read_text())
                self.assertEqual(report["pcl_types"], [view])


if __name__ == "__main__":
    unittest.main()
