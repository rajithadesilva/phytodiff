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
from tomato_recon.data.diffts import DiffTSInput, from_sample, load_top_down_input, rgb_available, sample_points
from tomato_recon.data.processed import make_tiny_sample
from tomato_recon.models.diffusion.diffts_model import Condition, Stage1ConditionAdapter, sparse_xyz
from tomato_recon.models.diffusion.diffts_sampling import (
    create_scheduler, direction_mst, q_sample, sample,
)
from tomato_recon.models.diffusion.diffts_runtime import (
    STAGE1_ADAPTER_SCHEMA, TRAIN_VIEWS, load_stage2_checkpoint,
    save_stage2_checkpoint, set_stage1_frozen, stage1_features,
)
from tomato_recon.models.diffusion.model import ConditionalSkeletonDenoiser


class DiffTSTests(unittest.TestCase):
    def test_stage2_config_and_colour_provenance(self) -> None:
        cfg, _ = load_config("diffusion", ["model.diffusion.variant=points"])
        self.assertEqual(list(cfg.data.pcl_types), ["full", "top_down", "side"])
        self.assertEqual(cfg.model.diffusion.variant, "points")
        self.assertTrue(cfg.trainer.freeze_stage1)
        self.assertEqual(float(cfg.trainer.stage1_learning_rate), 0.00001)
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

    def test_stage1_adapter_voxel_alignment_and_gradients(self) -> None:
        class Sparse:
            def __init__(self, features, indices, spatial_shape, batch_size):
                self.features, self.indices = features, indices

        xyz = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0.016, 0., 0.]])
        points = torch.randn(3, 2, requires_grad=True)
        global_feature = torch.randn(1, requires_grad=True)
        adapter = Stage1ConditionAdapter(2, 1, 4, 0.016)
        with patch("tomato_recon.models.diffusion.diffts_model.spconv",
                   SimpleNamespace(SparseConvTensor=Sparse)):
            field = adapter(xyz, points, global_feature)
        self.assertEqual(field.sparse.features.shape, (2, 4))
        torch.testing.assert_close(
            sparse_xyz(field.sparse, field.origin_xyz, field.resolution),
            torch.tensor([[0.008, 0.008, 0.008], [0.024, 0.008, 0.008]]),
        )
        projected = adapter.project(torch.cat(
            (points, global_feature.expand(3, -1)), -1
        ))
        torch.testing.assert_close(field.sparse.features[0],
                                   projected[:2].mean(0))
        field.sparse.features.sum().backward()
        self.assertIsNotNone(points.grad)
        self.assertIsNotNone(global_feature.grad)
        self.assertTrue(any(p.grad is not None for p in adapter.parameters()))

    def test_stage1_backbone_freeze_and_unfreeze(self) -> None:
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(6, 4)

            def forward(self, xyz, features, mask):
                points = self.projection(features)
                return points, [], points.mean(1)

        backbone = TinyBackbone()
        data = DiffTSInput(
            "plant", "combined", "full", torch.zeros(3, 3),
            torch.ones(3, 3), torch.ones(3, 3), True,
        )
        optimizer = torch.optim.AdamW([{"params": backbone.parameters(), "lr": 0.01}])
        original = backbone.projection.weight.detach().clone()
        set_stage1_frozen(backbone, True)
        frozen_points, _ = stage1_features(backbone, data, frozen=True)
        self.assertFalse(frozen_points.requires_grad)
        self.assertFalse(backbone.training)
        set_stage1_frozen(backbone, False)
        points, global_feature = stage1_features(backbone, data, frozen=False)
        (points.sum() + global_feature.sum()).backward()
        self.assertTrue(backbone.training)
        self.assertIsNotNone(backbone.projection.weight.grad)
        optimizer.step()
        self.assertFalse(torch.equal(original, backbone.projection.weight))
        optimizer.zero_grad(set_to_none=True)
        set_stage1_frozen(backbone, True)
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))

    def test_stage1_adapter_checkpoint_embeds_backbone(self) -> None:
        class TinyModel(torch.nn.Linear):
            variant = "points_stage1"
            max_nodes = 256

        cfg, _ = load_config("diffusion", ["model.diffusion.variant=points_stage1"])
        model = TinyModel(1, 1)
        backbone = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": 0.0001},
            {"params": backbone.parameters(), "lr": 0.00001},
        ])
        original = backbone.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.ckpt"
            save_stage2_checkpoint(
                path, model, optimizer, cfg, 3, {"best_val_skeleton_chamfer_m": 0.1},
                "source-hash", {"signature": "data-hash"}, backbone=backbone,
            )
            with torch.no_grad():
                backbone.weight.zero_()
            loaded = load_stage2_checkpoint(
                path, model, backbone=backbone, optimizer=optimizer,
                compatibility={"signature": "data-hash"},
            )
            self.assertEqual(loaded["schema_version"], STAGE1_ADAPTER_SCHEMA)
            self.assertEqual(loaded["stage1_checkpoint_sha256"], "source-hash")
            self.assertTrue(loaded["stage1_frozen"])
            torch.testing.assert_close(backbone.weight, original)

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
            torch.save({
                "schema_version": "diffts-stage2-v1", "variant": "points_stage1",
                "model_state": model.state_dict(),
            }, path)
            with self.assertRaisesRegex(ValueError, "old points_stage1 checkpoints"):
                load_stage2_checkpoint(path, model)
            torch.save({"stage": "diffusion", "model_state": model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "old stage 2 checkpoints"):
                load_stage2_checkpoint(path, model, stage1_hash=None)

    def test_legacy_model_is_retired(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "await DiffTS"):
            ConditionalSkeletonDenoiser()


if __name__ == "__main__":
    unittest.main()
