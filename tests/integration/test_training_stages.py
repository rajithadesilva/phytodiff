from __future__ import annotations

import unittest

from tomato_recon.train import train_graph, train_joint, train_parametric


class TrainingStageIntegrationTests(unittest.TestCase):
    def test_later_stages_report_pending_diffts_integration(self) -> None:
        for stage in (train_graph, train_parametric, train_joint):
            with self.subTest(stage=stage.__name__):
                with self.assertRaisesRegex(RuntimeError, "await DiffTS"):
                    stage.main([])


if __name__ == "__main__":
    unittest.main()
