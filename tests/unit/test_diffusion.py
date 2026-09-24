from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tomato_recon.config import load_config
from tomato_recon.data.diffts import from_sample, load_top_down_input, rgb_available, sample_points
from tomato_recon.data.processed import make_tiny_sample
from tomato_recon.models.diffusion.diffts_model import Condition
from tomato_recon.models.diffusion.diffts_sampling import (
    create_scheduler, direction_mst, q_sample, sample,
)
from tomato_recon.models.diffusion.diffts_runtime import load_stage2_checkpoint, TRAIN_VIEWS
from tomato_recon.models.diffusion.model import ConditionalSkeletonDenoiser


class DiffTSTests(unittest.TestCase):
    def test_stage2_config_and_colour_provenance(self) -> None:
        cfg, _ = load_config("diffusion")
        self.assertEqual(list(cfg.data.pcl_types), ["full", "top_down", "side"])
        self.assertEqual(cfg.model.diffusion.variant, "points")
        self.assertFalse(rgb_available({"dataset": "pheno4d", "rgb_available": False}))
        self.assertTrue(rgb_available({"dataset": "tomatopgt", "rgb_available": True}))
        self.assertTrue(rgb_available({"dataset": "tomatowur"}))

    def test_target_free_top_down_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "top_down.npz"
            metadata = {"dataset": "pheno4d", "rgb_available": False,
                        "top_down": {"algorithm_version": "xy-footprint-v1"}}
            np.savez(path, plant_id="plant_1", xyz=np.ones((3, 3), np.float32),
                     rgb=np.zeros((3, 3), np.float32), normals=np.zeros((3, 3), np.float32),
                     point_valid=np.array([True, True, False]),
                     metadata_json=json.dumps(metadata))
            sample = load_top_down_input(path)
            self.assertEqual(len(sample.xyz), 2)
            self.assertIsNone(sample.target_xyz)
            self.assertFalse(sample.rgb_available)

    def test_targets_and_point_selection(self) -> None:
        sample = from_sample(make_tiny_sample(16, 64))
        chosen = sample_points(sample, 16, seed=3)
        self.assertEqual(len(chosen.xyz), 16)
        self.assertEqual(len(chosen.target_xyz), int(make_tiny_sample(16, 64).node_valid.sum()))

    def test_noise_and_direction_tree(self) -> None:
        scheduler = create_scheduler(20)
        clean = torch.zeros(8, 6)
        noise = torch.ones_like(clean)
        noisy = q_sample(clean, torch.tensor([10]), noise, scheduler)
        self.assertEqual(noisy.shape, clean.shape)
        self.assertTrue(torch.isfinite(noisy).all())
        xyz = torch.tensor([[0., 0., 0.], [0., 0., 0.1], [0., 0., 0.2]])
        flow = torch.tensor([[0., 0., 0.], [0., 0., -1.], [0., 0., -1.]])
        parents, edges = direction_mst(xyz, flow)
        self.assertEqual(parents.tolist(), [-1, 0, 1])
        self.assertEqual(edges.shape, (2, 2))

    def test_sampling_callback_records_initial_and_each_denoising_state(self) -> None:
        class Sparse:
            features = torch.ones(2, 1)

            def replace_feature(self, features):
                result = Sparse()
                result.features = features
                return result

        class TinyModel:
            max_nodes = 2

            def denoise(self, state, timestep, condition):
                return torch.zeros_like(state)

        class Solver:
            config = SimpleNamespace(num_train_timesteps=10, beta_start=2e-6, beta_end=2e-4)
            alphas_cumprod = torch.linspace(0.99, 0.9, 10)

            def set_timesteps(self, steps, device):
                self.timesteps = torch.tensor([9, 5, 0][:steps], device=device)

            def step(self, predicted, timestep, residual, generator=None):
                return SimpleNamespace(prev_sample=residual * 0.5)

        condition = Condition(Sparse(), torch.zeros(3), 1.0, torch.tensor([1.0]))
        xyz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
        captured = []
        with patch("tomato_recon.models.diffusion.diffts_sampling.create_scheduler",
                   return_value=Solver()):
            prediction = sample(
                TinyModel(), condition, xyz, steps=3, guidance=1.0, seed=7,
                scheduler=Solver(),
                on_step=lambda index, timestep, state: captured.append(
                    (index, timestep, state.clone())
                ),
            )
        self.assertEqual([row[0] for row in captured], [0, 1, 2, 3])
        self.assertEqual([row[1] for row in captured], [9, 9, 5, 0])
        torch.testing.assert_close(captured[-1][2][:, :3], prediction.node_xyz)

    def test_checkpoint_rejects_variant_and_stage1_mismatch(self) -> None:
        class TinyModel(torch.nn.Linear):
            variant = "points"
            max_nodes = 256
        model = TinyModel(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.ckpt"
            torch.save({
                "schema_version": "diffts-stage2-v1", "variant": "points",
                "stage1_checkpoint_sha256": None, "max_nodes": 256,
                "pcl_types": list(TRAIN_VIEWS), "model_state": model.state_dict(),
            }, path)
            load_stage2_checkpoint(path, model, stage1_hash=None)
            model.variant = "points_stage1"
            with self.assertRaisesRegex(ValueError, "variant mismatch"):
                load_stage2_checkpoint(path, model, stage1_hash=None)
            model.variant = "points"
            with self.assertRaisesRegex(ValueError, "feature source mismatch"):
                load_stage2_checkpoint(path, model, stage1_hash="different")
            torch.save({"stage": "diffusion", "model_state": model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "old stage 2 checkpoints"):
                load_stage2_checkpoint(path, model, stage1_hash=None)

    def test_legacy_model_is_retired(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "await DiffTS"):
            ConditionalSkeletonDenoiser()


if __name__ == "__main__":
    unittest.main()
