from __future__ import annotations

import unittest

import torch

from tomato_recon.data.collate import collate_plant_samples
from tomato_recon.data.tomatowur import make_tiny_sample
from tomato_recon.models.diffusion.losses import masked_mse
from tomato_recon.models.diffusion.model import ConditionalSkeletonDenoiser
from tomato_recon.models.diffusion.sampling import existence_nms, sample_skeleton
from tomato_recon.models.diffusion.scheduler import DiffusionScheduler
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.pointnext import PointNeXtAdapter


class DiffusionTests(unittest.TestCase):
    def test_q_sample_and_reverse_shape(self) -> None:
        scheduler = DiffusionScheduler(20)
        clean = torch.randn(2, 8, 6)
        noise = torch.randn_like(clean)
        timestep = torch.tensor([3, 10])
        noisy = scheduler.q_sample(clean, timestep, noise)
        self.assertEqual(noisy.shape, clean.shape)
        reverse = scheduler.ddim_step(noise, timestep, 2, noisy)
        self.assertEqual(reverse.shape, clean.shape)

    def test_masked_loss(self) -> None:
        prediction = torch.tensor([[[1.0], [100.0]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(float(masked_mse(prediction, target, mask)), 1.0)

    def test_existence_pruning_and_nms(self) -> None:
        xyz = torch.tensor([[0.0, 0, 0], [0.001, 0, 0], [0.1, 0, 0]])
        probability = torch.tensor([0.9, 0.8, 0.7])
        keep = existence_nms(xyz, probability, 0.5, 0.005)
        self.assertEqual(int(keep.sum()), 2)
        self.assertTrue(keep[0] and keep[2])

    def test_seeded_sampling_and_flow_normalisation(self) -> None:
        torch.manual_seed(5)
        batch = collate_plant_samples([make_tiny_sample(16, 64)])
        encoder = PointEncoder(PointNeXtAdapter(6, 32, 48))
        encoded = encoder(batch.xyz, torch.cat([batch.rgb, batch.normals], -1), batch.point_valid)
        model = ConditionalSkeletonDenoiser(16, 32, 48, 32, 1, 4, 4)
        scheduler = DiffusionScheduler(20)
        first = sample_skeleton(model, scheduler, encoded, batch.point_valid, sample_steps=3, seed=11)
        second = sample_skeleton(model, scheduler, encoded, batch.point_valid, sample_steps=3, seed=11)
        torch.testing.assert_close(first.node_xyz, second.node_xyz)
        norms = first.parent_flow.norm(dim=-1)
        self.assertTrue(bool(((norms < 1e-6) | torch.isclose(norms, torch.ones_like(norms), atol=1e-5)).all()))


if __name__ == "__main__":
    unittest.main()

