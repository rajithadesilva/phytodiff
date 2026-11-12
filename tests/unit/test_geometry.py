from __future__ import annotations

import unittest

import torch

from tomato_recon.geometry.frames import parallel_transport_frames
from tomato_recon.geometry.fruits import fruit_ellipsoid
from tomato_recon.geometry.leaves import leaf_surface
from tomato_recon.geometry.spline import sample_cubic_bspline
from tomato_recon.geometry.tubes import stem_tube
from tomato_recon.evaluation.geometry_metrics import geometry_metrics
from tomato_recon.models.parametric.losses import occlusion_aware_geometry_loss


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.01, 0.05], [0.01, 0.01, 0.1], [0.02, 0.0, 0.15]]
        )

    def test_spline_endpoints_and_gradients(self) -> None:
        control = self.control.clone().requires_grad_()
        curve = sample_cubic_bspline(control, 12)
        torch.testing.assert_close(curve[0], control[0])
        torch.testing.assert_close(curve[-1], control[-1])
        curve.square().sum().backward()
        self.assertIsNotNone(control.grad)

    def test_parallel_transport_frames(self) -> None:
        curve = sample_cubic_bspline(self.control, 12)
        tangent, normal, binormal = parallel_transport_frames(curve)
        self.assertTrue(torch.isfinite(normal).all())
        torch.testing.assert_close((tangent * normal).sum(-1), torch.zeros(12), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(binormal.norm(dim=-1), torch.ones(12), atol=1e-5, rtol=1e-5)

    def test_tube_validity(self) -> None:
        mesh = stem_tube(self.control, 0.005, 0.002, curve_samples=8, radial_segments=6)
        mesh.validate()
        self.assertEqual(len(mesh.vertices), 48)
        self.assertGreater(len(mesh.faces), 0)

    def test_negative_radius_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stem_tube(self.control, -0.1, 0.002)

    def test_leaf_triangulation(self) -> None:
        mesh = leaf_surface(self.control, [0.0, 0.02, 0.0], [0.001, 0.0], curve_samples=8)
        mesh.validate()
        self.assertEqual(mesh.faces.shape, (14, 3))

    def test_fruit_ellipsoid(self) -> None:
        mesh = fruit_ellipsoid(torch.zeros(3), [0.02, 0.018, 0.025])
        mesh.validate()
        self.assertTrue(torch.isfinite(mesh.vertices).all())

    def test_geometry_metrics_and_occlusion_weights(self) -> None:
        points = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        metrics = geometry_metrics(
            points, points, model_normals=normals, scan_normals=-normals
        )
        self.assertEqual(metrics["bidirectional_chamfer_m2"], 0.0)
        self.assertEqual(metrics["normal_consistency"], 1.0)
        unsupported = torch.tensor([[1.0, 0.0, 0.0]])
        visible = occlusion_aware_geometry_loss(points, unsupported, torch.ones(1))
        hidden = occlusion_aware_geometry_loss(points, unsupported, torch.zeros(1))
        self.assertGreater(float(visible), float(hidden))


if __name__ == "__main__":
    unittest.main()
