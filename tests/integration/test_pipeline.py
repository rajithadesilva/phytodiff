from __future__ import annotations

import unittest

from tomato_recon.config import load_config
from tomato_recon.models.pipeline import TomatoReconstructionPipeline


class PipelineIntegrationTests(unittest.TestCase):
    def test_later_pipeline_reports_pending_diffts_integration(self) -> None:
        cfg, _ = load_config("infer")
        with self.assertRaisesRegex(RuntimeError, "await DiffTS"):
            TomatoReconstructionPipeline(cfg)


if __name__ == "__main__":
    unittest.main()
