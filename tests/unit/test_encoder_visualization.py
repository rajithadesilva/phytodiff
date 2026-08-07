from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from scripts.visualize_encoder_predictions import encoder_metrics, render_encoder_prediction
from tomato_recon.data.collate import collate_plant_samples
from tomato_recon.data.tomatowur import make_tiny_sample
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.pointnext import PointNeXtAdapter


class EncoderVisualizationTests(unittest.TestCase):
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
                skeleton_threshold_m=0.006,
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


if __name__ == "__main__":
    unittest.main()
