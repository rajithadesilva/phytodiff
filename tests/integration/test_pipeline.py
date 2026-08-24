from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tomato_recon.config import load_config
from tomato_recon.data.collate import collate_plant_samples
from tomato_recon.data.processed import make_tiny_sample
from tomato_recon.infer import write_inference_outputs
from tomato_recon.models.pipeline import TomatoReconstructionPipeline


class PipelineIntegrationTests(unittest.TestCase):
    def test_encoder_diffusion_graph_parametric_usd(self) -> None:
        cfg, _ = load_config("infer", ["--config", "configs/smoke/all.yaml"])
        sample = make_tiny_sample(16, 64)
        pipeline = TomatoReconstructionPipeline(cfg).eval()
        result = pipeline.reconstruct(collate_plant_samples([sample]))[0]
        result.graph.validate()
        result.geometry.validate()
        self.assertTrue(result.uncertainty["geometry_visibility_weight_map"])
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_inference_outputs(Path(directory), sample, result, cfg)
            expected = {
                "input_normalised.ply",
                "semantic_prediction.ply",
                "skeleton_prediction.ply",
                "skeleton_prediction.json",
                "plant_graph.json",
                "organ_parameters.json",
                "reconstructed_mesh.ply",
                "traits.json",
                "uncertainty.json",
                "plant.usd",
                "plant_debug.usda",
                "export_report.json",
                "preview.png",
            }
            self.assertTrue(expected.issubset(outputs))


if __name__ == "__main__":
    unittest.main()
