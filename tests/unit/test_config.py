from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tomato_recon.config import load_config
from tomato_recon.data.processed import make_tiny_sample
from tomato_recon.models.encoders.registry import create_backbone, ensure_backbone_available, list_backbones
from tomato_recon.train.common import load_checkpoint, save_checkpoint


class ConfigurationTests(unittest.TestCase):
    def test_kpconvx_is_the_default_stage1_encoder(self) -> None:
        cfg, _ = load_config("encoder")
        self.assertEqual(cfg.model.encoder.name, "kpconvx")
        self.assertFalse(cfg.model.encoder.pretrained)
        self.assertEqual(
            cfg.model.encoder.checkpoint, "outputs/stage1_ablation/kpconvx/best.ckpt"
        )

        diffusion_cfg, _ = load_config("diffusion")
        self.assertEqual(diffusion_cfg.model.encoder.name, "kpconvx")
        self.assertEqual(
            diffusion_cfg.model.encoder.checkpoint,
            "outputs/stage1_ablation/kpconvx/best.ckpt",
        )

    def test_stage1_registry_has_only_supported_models(self) -> None:
        self.assertEqual(set(list_backbones()), {"pointnext", "sonata_ptv3", "kpconvx"})
        for removed in ("ptv3", "litept"):
            with self.assertRaisesRegex(ValueError, "unknown point backbone"):
                ensure_backbone_available(removed)

    def test_pointnext_switch(self) -> None:
        cfg, _ = load_config("encoder", ["--config", "configs/smoke/all.yaml"])
        backbone = create_backbone(
            cfg.model.encoder.name,
            input_dim=6,
            output_dim=cfg.model.encoder.output_dim,
            global_dim=cfg.model.encoder.global_dim,
        )
        self.assertEqual(backbone.output_dim, 32)

    def test_optional_dependency_error_is_actionable(self) -> None:
        unavailable = next((name for name, ready in list_backbones().items() if not ready), None)
        if unavailable is None:
            self.skipTest("all optional backbones are installed")
        with self.assertRaisesRegex(ImportError, "requires optional"):
            ensure_backbone_available(unavailable)

    def test_checkpoint_hash_and_k_mismatch(self) -> None:
        cfg, _ = load_config("encoder", ["--config", "configs/smoke/all.yaml"])
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        sample = make_tiny_sample(16, 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ckpt"
            save_checkpoint(
                path,
                stage="test",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                sample=sample,
                epoch=0,
                metrics={},
            )
            with self.assertRaisesRegex(ValueError, "preprocessing hash mismatch"):
                load_checkpoint(path, model, expected_preprocessing_hash="different")
            with self.assertRaisesRegex(ValueError, "max_nodes mismatch"):
                load_checkpoint(path, model, expected_max_nodes=17)


if __name__ == "__main__":
    unittest.main()
