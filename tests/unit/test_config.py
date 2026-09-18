from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tomato_recon.config import load_config
from tomato_recon.data.processed import make_tiny_sample
from tomato_recon.models.encoders.registry import create_backbone, ensure_backbone_available, list_backbones
from tomato_recon.train.common import (
    dataset_compatibility_contains,
    load_checkpoint,
    save_checkpoint,
)


class ConfigurationTests(unittest.TestCase):
    def test_kpconvx_is_the_default_stage1_encoder(self) -> None:
        cfg, _ = load_config("encoder")
        self.assertEqual(cfg.model.encoder.name, "kpconvx")
        self.assertFalse(cfg.model.encoder.pretrained)
        self.assertEqual(cfg.model.encoder.skeleton_threshold_m, 0.01)
        self.assertEqual(cfg.model.encoder.junction_threshold_multiplier, 2.0)
        self.assertEqual(
            cfg.model.encoder.checkpoint,
            "outputs/stage1_ablation_1/combined/kpconvx/best.ckpt",
        )
        self.assertEqual(cfg.data.dataset, "combined")
        self.assertEqual(list(cfg.data.pcl_types), ["full"])

        for pcl_types in ("[top_down]", "[side]", "[full,top_down,side]"):
            selected, _ = load_config("encoder", [f"data.pcl_types={pcl_types}"])
            self.assertEqual(
                list(selected.data.pcl_types), pcl_types.strip("[]").split(",")
            )
        for invalid in ("[]", "[full,full]", "[partial]", "full"):
            with self.assertRaisesRegex(ValueError, "data.pcl_types"):
                load_config("encoder", [f"data.pcl_types={invalid}"])
        with self.assertRaisesRegex(ValueError, "data.pcl_type"):
            load_config("encoder", ["data.pcl_type=top_down"])

        diffusion_cfg, _ = load_config("diffusion")
        self.assertEqual(diffusion_cfg.model.encoder.name, "kpconvx")
        self.assertEqual(
            diffusion_cfg.model.encoder.checkpoint,
            "outputs/stage1_ablation_1/combined/kpconvx/best.ckpt",
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
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("dataset_compatibility", checkpoint)
            self.assertTrue(checkpoint["dataset_compatibility"]["signature"])
            self.assertNotIn("pcl_type", checkpoint)
            self.assertEqual(checkpoint["pcl_types"], ["full"])
            with self.assertRaisesRegex(ValueError, "preprocessing hash mismatch"):
                load_checkpoint(path, model, expected_preprocessing_hash="different")
            with self.assertRaisesRegex(ValueError, "max_nodes mismatch"):
                load_checkpoint(path, model, expected_max_nodes=17)
            with self.assertRaisesRegex(ValueError, "point-cloud types mismatch"):
                load_checkpoint(path, model, expected_pcl_types=["top_down"])
            with self.assertRaisesRegex(ValueError, "point-cloud types mismatch"):
                load_checkpoint(path, model, expected_pcl_types=["side", "full"])
            legacy = dict(checkpoint)
            legacy.pop("pcl_types")
            legacy["config"] = {
                **legacy["config"],
                "data": {**legacy["config"]["data"], "pcl_type": "top_down"},
            }
            legacy["config"]["data"].pop("pcl_types", None)
            torch.save(legacy, path)
            load_checkpoint(path, model, expected_pcl_types=["top_down"])
            with self.assertRaisesRegex(ValueError, "point-cloud types mismatch"):
                load_checkpoint(path, model, expected_pcl_types=["full"])
            checkpoint = legacy
            expected = checkpoint["dataset_compatibility"]
            checkpoint["dataset_compatibility"] = {
                **expected,
                "datasets": {
                    **expected["datasets"],
                    "another-source": ["another-hash"],
                },
                "signature": "combined-signature",
            }
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "dataset compatibility mismatch"):
                load_checkpoint(path, model, expected_dataset_compatibility=expected)
            load_checkpoint(
                path,
                model,
                expected_dataset_compatibility=expected,
                allow_dataset_subset=True,
            )

    def test_combined_checkpoint_compatibility_contains_source_evaluation(self) -> None:
        shared = {
            "schema_version": "1.0",
            "manifest_schema_version": "1.0",
            "layout": "flat-plant-instance-v1",
        }
        combined = {
            **shared,
            "datasets": {
                "tomatowur": ["wur-hash"],
                "tomatopgt": ["pgt-hash"],
                "pheno4d": ["pheno-hash"],
            },
        }
        pheno4d = {**shared, "datasets": {"pheno4d": ["pheno-hash"]}}
        wrong_pheno4d = {**shared, "datasets": {"pheno4d": ["different-hash"]}}
        self.assertTrue(dataset_compatibility_contains(combined, pheno4d))
        self.assertFalse(dataset_compatibility_contains(pheno4d, combined))
        self.assertFalse(dataset_compatibility_contains(combined, wrong_pheno4d))


if __name__ == "__main__":
    unittest.main()
