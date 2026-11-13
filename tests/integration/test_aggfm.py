from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from tests.fixtures.aggfm import save_stage1, tiny_config, write_dataset
from tomato_recon.data.aggfm import from_sample
from tomato_recon.data.processed import make_tiny_sample, processed_dataset_compatibility
from tomato_recon.data.schemas import PlantGraph
from tomato_recon.evaluate_aggfm import evaluate
from tomato_recon.infer_aggfm import run
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.models.flow_matching.losses import training_loss
from tomato_recon.models.flow_matching.runtime import create_model, extract_stage1, predict
from tomato_recon.train.train_aggfm import main as train
from tomato_recon.visualize_aggfm import visualize


class AGGFMIntegrationTests(unittest.TestCase):
    def test_cpu_train_resume_evaluate_target_free_infer_visualize(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                write_dataset(data_root)
                cfg = tiny_config("data.dataset=combined", "data.pcl_types=[top_down]",
                                  "trainer.fast_dev_run=true", "trainer.devices=0")
                cfg.data.processed_root = str(data_root)
                cfg.output.dir = str(root / "run")
                cfg.model.encoder.checkpoint = str(root / "stage1.ckpt")
                encoder = PointEncoder(create_backbone_from_config(cfg.model.encoder))
                compatibility = processed_dataset_compatibility(data_root, "combined")
                save_stage1(cfg.model.encoder.checkpoint, encoder, cfg, compatibility)
                config_path = root / "config.yaml"
                OmegaConf.save(cfg, config_path)
                train(["--config", str(config_path)])
                last = root / "run/last.ckpt"
                self.assertTrue((root / "run/best.ckpt").is_file())
                train(["--config", str(config_path), "--resume", str(last)])
                payload = torch.load(last, weights_only=False)
                self.assertEqual(payload["epoch"], 1)
                # Embedded complete Stage 1 state must suffice after removing its source.
                Path(cfg.model.encoder.checkpoint).unlink()
                report = evaluate(last, root / "test.json", dataset="tomatowur", views=["side"], steps=1)
                self.assertEqual(report["sample_count"], 1)
                self.assertEqual(report["overall"]["tree_validity"], 1.)
                result = run(last, data_root / "plant_000003/side.npz", root / "prediction", view="side", steps=1)
                self.assertGreaterEqual(result["predicted_count"], 1)
                self.assertTrue((root / "prediction/graph.json").is_file())
                progress = io.StringIO()
                with redirect_stdout(progress):
                    visualize(last, root / "viz", views=["full", "side"], steps=2, frames=3, count=0)
                expected_files = set()
                for plant in ("plant_000003", "plant_000006", "plant_000009"):
                    for view in ("full", "side"):
                        folder = root / "viz/test"
                        image_path = folder / "images" / view / f"{plant}.png"
                        graph_path = folder / "graphs" / view / f"{plant}.json"
                        trajectory_path = folder / "trajectories" / view / f"{plant}.npz"
                        expected_files.update((image_path, graph_path, trajectory_path))
                        with Image.open(image_path) as image:
                            image.verify()
                        graph = PlantGraph.from_dict(json.loads(graph_path.read_text()))
                        self.assertEqual(graph.plant_id, plant)
                        self.assertEqual(graph.source["view"], view)
                        with np.load(trajectory_path) as trajectory:
                            np.testing.assert_allclose(trajectory["times"], [0., 0.5, 1.])
                            self.assertEqual(trajectory["states"].shape, (3, 16, 4))
                            self.assertTrue(np.isfinite(trajectory["states"]).all())
                self.assertEqual({p for p in (root / "viz").rglob("*") if p.is_file()}, expected_files)
                log = progress.getvalue()
                self.assertIn("Loading checkpoint", log)
                self.assertIn("3 plants x 2 views = 6 samples", log)
                self.assertIn("Sample 6/6 | plant_000009 | pheno4d/side", log)
                self.assertEqual(log.count("Flow step 0/2"), 6)
                self.assertEqual(log.count("Flow step 2/2"), 6)
                self.assertIn("Saved image", log)
                self.assertIn("Saved graph", log)
                self.assertIn("Saved trajectory", log)
                self.assertIn("Complete 6/6 (100.0%)", log)
                self.assertIn("ETA 0:00:00", log)
                self.assertIn("Complete | 6 images, 6 graphs, 6 trajectories", log)
                # A limited validation preview has separate paths and uses saved flow steps.
                with redirect_stdout(io.StringIO()):
                    visualize(last, root / "viz", split="val", views=["side"], frames=3, count=1)
                self.assertEqual(list((root / "viz/val/images/side").iterdir()),
                                 [root / "viz/val/images/side/plant_000002.png"])
        finally:
            torch.set_num_threads(previous_threads)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA; run in the Docker train service")
    def test_cuda_forward_backward_update_and_inference_kpconvx(self):
        cfg = tiny_config("model.encoder.name=kpconvx", "model.encoder.layer_blocks=[1,1]",
                          "model.encoder.neighbor_limits=[8,8]", "model.encoder.init_channels=16")
        data = from_sample(make_tiny_sample(16, 64)).to("cuda")
        encoder = PointEncoder(create_backbone_from_config(cfg.model.encoder)).cuda()
        model = create_model(cfg).cuda()
        # Fine-tuning smoke exercises gradients through all Stage 1 prediction heads.
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(encoder.parameters()), lr=1e-4)
        before = model.field.velocity.weight.detach().clone()
        seeds, memory = model.condition(extract_stage1(encoder, data, frozen=False))
        losses, _ = training_loss(model, seeds, memory, data, cfg.model.aggfm)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        for module in (model, encoder):
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.field.velocity.weight))
        prediction, _ = predict(model, encoder, data.points_only(), cfg)
        self.assertTrue(torch.isfinite(prediction.node_xyz).all())
        prediction.graph.validate()


if __name__ == "__main__":
    unittest.main()
