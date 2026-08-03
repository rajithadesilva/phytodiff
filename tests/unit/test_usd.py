from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tomato_recon.data.tomatowur import make_tiny_sample
from tomato_recon.export.usd_exporter import USDPlantExporter, validate_usd_static
from tomato_recon.models.parametric.decoder import ParametricDecoder
from tomato_recon.models.parametric.primitives import generate_plant_geometry


class USDTests(unittest.TestCase):
    def test_stage_hierarchy_units_mesh_and_metadata(self) -> None:
        sample = make_tiny_sample(16, 64)
        decoder = ParametricDecoder(8, 16)
        parameters = decoder.decode_graph(sample.graph_target, decoder(torch.zeros(1, 16, 8)))
        geometry = generate_plant_geometry(parameters, curve_samples=6, radial_segments=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plant.usd"
            report = USDPlantExporter().export(
                sample.graph_target,
                geometry,
                path,
                write_debug_usda=True,
                include_skeleton=True,
                include_collisions=False,
                include_context_pole=False,
            )
            self.assertTrue(report.valid)
            self.assertGreater(report.mesh_count, 0)
            self.assertTrue(path.is_file())
            self.assertTrue(path.with_name("plant_debug.usda").is_file())
            static = validate_usd_static(path)
            self.assertTrue(static["valid"])


if __name__ == "__main__":
    unittest.main()

