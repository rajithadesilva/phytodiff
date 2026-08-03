from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TrainingStageIntegrationTests(unittest.TestCase):
    def test_every_stage_writes_checkpoint_and_resumes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        stages = ["encoder", "diffusion", "graph", "parametric", "joint"]
        with tempfile.TemporaryDirectory() as directory:
            for stage in stages:
                output = Path(directory) / stage
                command = [
                    sys.executable,
                    "-m",
                    f"tomato_recon.train.train_{stage}",
                    "--config",
                    "configs/smoke/all.yaml",
                    f"output.dir={output}",
                ]
                subprocess.run(command, cwd=root, env=environment, check=True, capture_output=True, text=True)
                checkpoint = output / "best.ckpt"
                self.assertTrue(checkpoint.is_file(), stage)
                self.assertTrue((output / "resolved_config.yaml").is_file(), stage)
                if stage == "joint":
                    self.assertTrue((output / "smoke_visibility_weights.pt").is_file())
                    self.assertTrue((output / "staged_validation.json").is_file())
                resumed = command + ["--resume", str(checkpoint)]
                subprocess.run(
                    resumed,
                    cwd=root,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
