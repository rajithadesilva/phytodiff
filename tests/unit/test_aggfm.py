from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf
import torch

from tests.fixtures.aggfm import save_stage1, tiny_config, write_dataset
from tomato_recon.config import repository_root
from tomato_recon.data.aggfm import from_sample, load_input
from tomato_recon.data.processed import PointCloudViewDataset, make_tiny_sample
from tomato_recon.evaluation.aggfm import metrics
from tomato_recon.models.encoders.base import PointEncoder
from tomato_recon.models.encoders.registry import create_backbone_from_config
from tomato_recon.models.flow_matching.flow import integrate, interpolate, supervised_endpoint
from tomato_recon.models.flow_matching.losses import parent_targets, training_loss
from tomato_recon.models.flow_matching.matching import Match, match_slots, target_structure
from tomato_recon.models.flow_matching.runtime import (
    create_model, create_stage1, extract_stage1, load_checkpoint, predict, save_checkpoint,
)
from tomato_recon.models.flow_matching.seeds import diverse_indices
from tomato_recon.models.flow_matching.topology import Prediction, decode, rooted_tree
from tomato_recon.models.flow_matching.transformer import parent_mask, parent_probabilities
from tomato_recon.models.stage2 import BACKENDS
from tomato_recon.train.train_aggfm import check_split_isolation, validation_view


class AGGFMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(2)
        self.cfg = tiny_config()
        self.data = from_sample(make_tiny_sample(16, 64))
        self.encoder = PointEncoder(create_backbone_from_config(self.cfg.model.encoder))
        self.model = create_model(self.cfg)

    def condition(self):
        return self.model.condition(extract_stage1(self.encoder, self.data))

    def test_seed_shapes_offsets_diversity_and_gradients(self):
        output = extract_stage1(self.encoder, self.data)
        output.centreline_offset[:] = 0.03
        seeds, memory = self.model.condition(output)
        self.assertEqual(seeds.state.shape, (16, 4))
        self.assertEqual(seeds.features.shape, (16, 32))
        self.assertEqual(memory.shape, (12, 32))
        self.assertEqual(int(seeds.observed.sum()), 12)
        shifted = self.data.xyz + 0.03
        self.assertLess(float(torch.cdist(seeds.state[:12, :3], shifted).min(-1).values.max()), 1e-5)
        indices = diverse_indices(shifted, shifted.new_ones(len(shifted)), 12)
        self.assertEqual(len(indices.unique()), 12)
        self.assertEqual(len(diverse_indices(torch.ones(5, 3), torch.ones(5), 5)), 1)
        seeds.state.sum().backward()
        self.assertGreater(float(self.model.seeds.latent_xyz.weight.grad.abs().sum()), 0)

    def test_frozen_and_finetuned_complete_stage1_outputs(self):
        output = extract_stage1(self.encoder, self.data)
        self.assertFalse(output.point_features.requires_grad)
        self.assertEqual(output.semantic_logits.shape, (1, 64, 5))
        for key in ("skeleton_logits", "junction_logits"):
            self.assertEqual(getattr(output, key).shape, (1, 64, 1))
        output = extract_stage1(self.encoder, self.data, frozen=False)
        output.centreline_offset.sum().backward()
        self.assertIsNotNone(self.encoder.offset_head[0].weight.grad)

    def test_complete_stage1_checkpoint_and_compatibility(self):
        compatibility = {"datasets": {"tomatowur": ["v1"]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.ckpt"
            self.cfg.model.encoder.checkpoint = str(path)
            save_stage1(path, self.encoder, self.cfg, compatibility)
            encoder, digest = create_stage1(self.cfg, "cpu", compatibility=compatibility)
            self.assertEqual(len(digest), 64)
            for key, tensor in self.encoder.state_dict().items():
                torch.testing.assert_close(tensor, encoder.state_dict()[key])
            bad = OmegaConf.create(OmegaConf.to_container(self.cfg))
            bad.model.encoder.output_dim = 32
            with self.assertRaisesRegex(ValueError, "architecture"):
                create_stage1(bad, "cpu")
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                create_stage1(self.cfg, "cpu", compatibility={"datasets": {"tomatowur": ["wrong"]}})
            payload = torch.load(path, weights_only=False)
            del payload["model_state"]["junction_head.weight"]
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                create_stage1(self.cfg, "cpu")

    def test_variable_candidate_target_counts_and_dummy_assignments(self):
        seeds, _ = self.condition()
        for n, m in ((3, 9), (16, 4), (1, 1)):
            limited = replace(seeds, state=seeds.state[:n], features=seeds.features[:n],
                              organ_logits=seeds.organ_logits[:n], structure=seeds.structure[:n],
                              observed=seeds.observed[:n])
            target = replace(self.data, target_xyz=self.data.target_xyz[:m],
                             target_parent=self.data.target_parent[:m],
                             organ_type=self.data.organ_type[:m], topology_role=self.data.topology_role[:m])
            match = match_slots(limited, target, self.cfg.model.aggfm.matching, 0.1)
            self.assertEqual(len(match.source.unique()), len(match.source))
            self.assertEqual(len(match.target.unique()), len(match.target))
            self.assertLessEqual(len(match.source), min(n, m))
        far = replace(seeds, state=seeds.state.detach().clone())
        far.state[:, :3] = 100
        match = match_slots(far, self.data, self.cfg.model.aggfm.matching, 0.1)
        self.assertEqual(len(match.source), 0)
        self.assertEqual(match.diagnostics["unmatched_targets"], 9)

    def test_structure_cost_changes_assignment(self):
        seeds, _ = self.condition()
        data = replace(self.data, target_xyz=torch.zeros(2, 3), target_parent=torch.tensor([-1, 0]),
                       topology_role=torch.tensor([0, 2]), organ_type=None)
        seeds = replace(seeds, state=torch.zeros(2, 4), structure=torch.tensor([[0., 1.], [0., 0.]]))
        weights = OmegaConf.create(dict(position=0., semantic=0., structure=1., unmatched_cost=4.))
        match = match_slots(seeds, data, weights, 0.1)
        self.assertEqual(dict(zip(match.source.tolist(), match.target.tolist())), {0: 1, 1: 0})
        self.assertEqual(match.diagnostics["structure"], 0.)

    def test_target_root_path_lengths_arbitrary_order(self):
        xyz = torch.tensor([[0., 0., 2.], [0., 0., 0.], [0., 0., 1.]])
        descriptor = target_structure(xyz, torch.tensor([2, -1, 1]), None, 1.)
        torch.testing.assert_close(descriptor[:, 0], torch.log1p(torch.tensor([2., 0., 1.])))
        with self.assertRaisesRegex(ValueError, "cycle"):
            target_structure(xyz, torch.tensor([1, 0, 1]), None, 1.)

    def test_flow_interpolation_velocity_and_existence_transport(self):
        seeds, _ = self.condition()
        match = Match(torch.tensor([0, 2]), torch.tensor([1, 3]), {})
        endpoint = supervised_endpoint(seeds, self.data, match, self.cfg.model.aggfm)
        self.assertEqual(float(endpoint[0, 3]), 3.)
        self.assertEqual(float(endpoint[1, 3]), -3.)
        torch.testing.assert_close(endpoint[1, :3], seeds.state[1, :3])
        for time in (0., 0.25, 1.):
            state, velocity = interpolate(seeds.state, endpoint, time)
            torch.testing.assert_close(velocity, endpoint - seeds.state)
            torch.testing.assert_close(state, seeds.state + time * velocity)

    def test_parent_targets_ignore_root_unmatched_and_missing_parents(self):
        match = Match(torch.tensor([3, 0, 2]), torch.tensor([0, 1, 3]), {})
        labels = parent_targets(5, self.data, match)
        self.assertEqual(labels.tolist(), [3, -100, -100, -100, -100])
        logits = torch.randn(5, 5)
        active = torch.tensor([True, False, True, True, False])
        valid = parent_mask(logits, active)
        probability = parent_probabilities(logits, active)
        self.assertFalse(valid.diag().any())
        self.assertEqual(float(probability[~valid].sum()), 0.)
        self.assertTrue(torch.isfinite(probability).all())
        self.assertEqual(float(parent_probabilities(torch.zeros(1, 1)).sum()), 0.)

    def test_empty_matching_losses_are_finite(self):
        seeds, memory = self.condition()
        data = replace(self.data, target_xyz=self.data.target_xyz + 100)
        losses, match = training_loss(self.model, seeds, memory, data, self.cfg.model.aggfm)
        self.assertEqual(match["matched"], 0.)
        for key in ("parent", "role", "geom"):
            self.assertEqual(float(losses[key]), 0.)
        losses["loss"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_training_update_and_inference_cpu(self):
        seeds, memory = self.condition()
        losses, match = training_loss(self.model, seeds, memory, self.data, self.cfg.model.aggfm, time=0.4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        before = self.model.field.velocity.weight.detach().clone()
        losses["loss"].backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.field.velocity.weight))
        prediction, _ = predict(self.model, self.encoder, self.data.points_only(), self.cfg)
        prediction.graph.validate()
        self.assertEqual(len(prediction.edges), prediction.predicted_count - 1)
        self.assertIn("structure", match)

    def test_seed_extremes_and_single_slot(self):
        for observed, latent in ((16, 0), (0, 16), (0, 1)):
            cfg = tiny_config(f"model.aggfm.observed_seeds={observed}", f"model.aggfm.latent_seeds={latent}")
            model = create_model(cfg)
            prediction, seeds = predict(model, self.encoder, self.data.points_only(), cfg)
            self.assertEqual(len(seeds.state), observed + latent)
            self.assertTrue(torch.isfinite(prediction.node_xyz).all())
            prediction.graph.validate()

    def test_ode_solvers_integrate_constant_velocity(self):
        seeds, memory = self.condition()
        def field(z, t, s, m):
            return {"velocity": torch.ones_like(z)}
        for solver in ("euler", "heun"):
            result, _ = integrate(field, seeds, memory, steps=3, solver=solver)
            torch.testing.assert_close(result, seeds.state + 1)

    def test_shared_block_geometry_preserves_outputs_and_position_gradients(self):
        block = self.model.field.blocks[0].eval()
        h = torch.randn(6, 32, requires_grad=True)
        xyz = torch.randn(6, 3, requires_grad=True)
        memory = torch.randn(8, 32, requires_grad=True)
        existence = torch.randn(6, requires_grad=True)
        inputs = (h, xyz, memory, existence)
        reference = block(*inputs)
        reference_gradients = torch.autograd.grad(reference.square().mean(), inputs)
        distance = torch.cdist(xyz.float(), xyz.float())
        local_distance = distance.detach().clone()
        local_distance.fill_diagonal_(torch.inf)
        neighbors = local_distance.topk(min(block.neighbors, len(h) - 1), largest=False).indices
        result = block(*inputs, distance=distance, neighbor_indices=neighbors)
        gradients = torch.autograd.grad(result.square().mean(), inputs)
        torch.testing.assert_close(result, reference)
        for gradient, reference_gradient in zip(gradients, reference_gradients):
            torch.testing.assert_close(gradient, reference_gradient, rtol=1e-4, atol=1e-6)

    def test_ode_velocity_path_matches_full_heads_and_final_predictions(self):
        self.model.eval()
        seeds, memory = self.condition()

        def full_field(state, time, seeds, memory):
            return self.model(state, time, seeds, memory)

        for solver in ("euler", "heun"):
            reference, reference_heads = integrate(full_field, seeds, memory, steps=3, solver=solver)
            with patch.object(self.model.field.parent, "forward",
                              wraps=self.model.field.parent.forward) as final_parent:
                result, heads = integrate(self.model, seeds, memory, steps=3, solver=solver)
            self.assertEqual(final_parent.call_count, 1)
            torch.testing.assert_close(result, reference)
            self.assertEqual(set(heads), set(reference_heads))
            for key in heads:
                torch.testing.assert_close(heads[key], reference_heads[key])

    def test_directed_tree_breaks_cycles_and_maximizes_weight(self):
        import itertools
        logits = torch.tensor([[-9., 9., 9.], [1., -9., 6.], [2., 5., -9.]])
        parent, edges = rooted_tree(logits, 0)
        self.assertEqual(parent[0], -1)
        self.assertEqual(len(edges), 2)
        options = []
        for p1, p2 in itertools.product((0, 2), (0, 1)):
            if not (p1 == 2 and p2 == 1):
                options.append(float(logits[1, p1] + logits[2, p2]))
        self.assertEqual(float(logits[1, parent[1]] + logits[2, parent[2]]), max(options))

    def test_tree_decoder_filters_nonfinite_scores_and_preserves_device(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            logits = torch.tensor([[-9., 1., 1.], [4., -9., float("nan")],
                                   [float("inf"), 2., -9.]], device=device)
            parents, edges = rooted_tree(logits, 0)
            self.assertEqual(parents.tolist(), [-1, 0, 1])
            self.assertEqual(edges.tolist(), [[0, 1], [1, 2]])
            self.assertEqual(parents.device, logits.device)
            self.assertEqual(edges.device, logits.device)
            allowed = torch.ones_like(logits, dtype=torch.bool)
            parents, masked_edges = rooted_tree(logits, 0, allowed)
            self.assertEqual(parents.tolist(), [-1, 0, 1])
            torch.testing.assert_close(masked_edges, edges)

    def test_decode_masks_inactive_and_relaxes_impossible_edge_bound(self):
        seeds, memory = self.condition()
        output = self.model(seeds.state, torch.tensor(1.), seeds, memory)
        state = seeds.state.detach().clone()
        state[:, 3] = -10
        state[:3, 3] = 10
        self.cfg.inference.max_edge_length_m = 1e-9
        prediction = decode(state, output, seeds, "test", self.cfg.inference)
        self.assertEqual(prediction.source_slots.tolist(), [0, 1, 2])
        self.assertTrue(prediction.graph.postprocessing[0]["edge_length_constraint_relaxed"])
        prediction.graph.validate()

    def test_inference_deterministic_and_ignores_targets(self):
        p1, _ = predict(self.model, self.encoder, self.data.points_only(), self.cfg)
        poisoned = replace(self.data, target_xyz=torch.full_like(self.data.target_xyz, float("nan")),
                           target_parent=torch.full_like(self.data.target_parent, 9999))
        p2, _ = predict(self.model, self.encoder, poisoned, self.cfg)
        torch.testing.assert_close(p1.node_xyz, p2.node_xyz)
        torch.testing.assert_close(p1.parent_index, p2.parent_index)

    def test_point_only_loader_does_not_access_target_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full.npz"
            np.savez(path, plant_id="plant_1", xyz=self.data.xyz.numpy(), rgb=self.data.rgb.numpy(),
                     normals=self.data.normals.numpy(), point_valid=np.ones(64, dtype=bool),
                     metadata_json=json.dumps(dict(dataset="pheno4d", rgb_available=False)),
                     node_xyz=np.array([object()], dtype=object))  # raises if read with allow_pickle=False
            result = load_input(path)
            self.assertIsNone(result.target_xyz)
            self.assertFalse(result.rgb_available)
            self.assertEqual(float(result.rgb.abs().sum()), 0.)

    def test_checkpoint_restores_optimizer_scheduler_rng_and_rejects_mismatch(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        seeds, memory = self.condition()
        training_loss(self.model, seeds, memory, self.data, self.cfg.model.aggfm)[0]["loss"].backward()
        optimizer.step()
        compatibility = {"signature": "v1", "datasets": {"tomatowur": ["v1"]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.ckpt"
            save_checkpoint(path, self.model, self.encoder, optimizer, self.cfg, 2, {}, "a" * 64,
                            compatibility, scheduler=scheduler)
            expected = torch.rand(4)
            original = self.model.field.velocity.weight.detach().clone()
            with torch.no_grad():
                self.model.field.velocity.weight.zero_()
            checkpoint = load_checkpoint(path, self.model, self.encoder, self.cfg, optimizer=optimizer,
                                          scheduler=scheduler, compatibility=compatibility, resume=True)
            torch.testing.assert_close(torch.rand(4), expected)
            torch.testing.assert_close(original, self.model.field.velocity.weight)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertTrue(optimizer.state)
            with self.assertRaisesRegex(ValueError, "compatibility"):
                load_checkpoint(path, self.model, self.encoder, self.cfg, compatibility={"signature": "v2"})
            with self.assertRaisesRegex(ValueError, "source"):
                load_checkpoint(path, self.model, self.encoder, self.cfg, stage1_hash="wrong")
            self.cfg.data.pcl_types = ["side"]
            with self.assertRaisesRegex(ValueError, "selection"):
                load_checkpoint(path, self.model, self.encoder, self.cfg)

    def test_datasets_all_nonempty_view_combinations_and_full_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root)
            for source in ("tomatowur", "tomatopgt", "pheno4d", "combined"):
                for size in (1, 2, 3):
                    for views in combinations(("full", "top_down", "side"), size):
                        dataset = PointCloudViewDataset(root, split="val", dataset=source, pcl_types=views)
                        self.assertEqual(len(dataset), (3 if source == "combined" else 1) * len(views))
                        first = from_sample(dataset[0])
                        for index in range(len(views)):
                            data = from_sample(dataset[index])
                            torch.testing.assert_close(data.target_xyz, first.target_xyz)
                            self.assertEqual(data.view, views[index])
                            self.assertEqual(len(data.organ_type), 9)
            for view in ("full", "top_down", "side"):
                data = load_input(root / "plant_000007" / f"{view}.npz", view)
                self.assertFalse(data.rgb_available)
                self.assertIsNone(data.target_xyz)

    def test_validation_selection_and_plant_split_isolation(self):
        self.assertEqual(validation_view(self.cfg), "top_down")
        self.cfg.data.pcl_types = ["side", "full"]
        self.assertEqual(validation_view(self.cfg), "side")
        self.cfg.trainer.validation_view = "full"
        self.assertEqual(validation_view(self.cfg), "full")
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset(Path(tmp))
            train = PointCloudViewDataset(tmp, split="train")
            val = PointCloudViewDataset(tmp, split="val")
            check_split_isolation(train, val)
            entries = train.full_dataset.manifest["instances"]
            entries[1]["source_plant_id"] = entries[0]["source_plant_id"]
            with self.assertRaisesRegex(ValueError, "leakage"):
                check_split_isolation(train, val)

    def test_perfect_metrics_after_permuting_node_ids(self):
        from tomato_recon.data.schemas import PlantGraph
        graph = PlantGraph.from_dict(make_tiny_sample().graph_target.to_dict())
        xyz = self.data.target_xyz
        permutation = torch.arange(len(xyz) - 1, -1, -1)
        inverse = permutation.argsort()
        parent = self.data.target_parent[permutation]
        parent = torch.where(parent < 0, -1, inverse[parent.clamp_min(0)])
        children = (parent >= 0).nonzero().flatten()
        edges = torch.stack((parent[children], children), -1)
        graph.root_node_id = int(inverse[0])
        # Graph is only used for validity; relabel its nodes/edges consistently.
        for node in graph.nodes:
            node.id = int(inverse[node.id])
        for edge in graph.edges:
            edge.parent, edge.child = int(inverse[edge.parent]), int(inverse[edge.child])
        prediction = Prediction(xyz[permutation], torch.zeros_like(xyz), parent, edges, permutation, graph)
        result = metrics(prediction, self.data)
        self.assertAlmostEqual(result["node_chamfer_m"], 0., places=6)
        self.assertEqual(result["parent_accuracy"], 1.)
        self.assertEqual(result["root_accuracy"], 1.)
        self.assertEqual(result["junction_f1"], 1.)

    def test_legacy_make_recipes_unchanged_and_registry_lazy(self):
        self.assertEqual(set(BACKENDS), {"diffts", "aggfm"})
        root = repository_root()
        default = subprocess.check_output(["make", "-n", "train-stage2"], cwd=root, text=True)
        self.assertEqual(default.count("-m tomato_recon.train.train_diffusion"), 2)
        self.assertIn("model.diffusion.variant=points_stage1", default)
        self.assertIn("output.dir=outputs/diffts/points_stage1_adapter", default)
        legacy_vis = subprocess.check_output(["make", "-n", "visualize-stage2-test"], cwd=root, text=True)
        self.assertIn("scripts/visualize_stage2_predictions.py", legacy_vis)
        for source in ("tomatowur", "tomatopgt", "pheno4d", "combined"):
            result = subprocess.check_output(["make", "-n", "train-stage2", "STAGE2_MODEL=aggfm",
                                              f"STAGE2_DATASET={source}", "STAGE2_PCL_TYPES=side"], cwd=root, text=True)
            self.assertIn(f"data.dataset={source}", result)
            self.assertIn("data.pcl_types=[side]", result)
            self.assertIn("docker compose", result)

    def test_invalid_configuration(self):
        for override in ("model.aggfm.observed_seeds=256", "model.aggfm.hidden_dim=31",
                         "model.aggfm.latent_noise_m=-1", "inference.flow_steps=0", "data.pcl_types=[]"):
            with self.assertRaises(ValueError):
                tiny_config(override)

    def test_selected_dispatch_preserves_legacy_view_fanout_and_split_checks(self):
        from tomato_recon.stage2 import main
        render = Mock()
        with patch("tomato_recon.stage2.runpy.run_path", return_value={"main": render}):
            main(["visualize", "--model", "diffts", "--pcl-types", "full", "side", "--split", "test",
                  "--checkpoint", "legacy.ckpt", "--output", "preview"])
            self.assertEqual(render.call_count, 2)
            self.assertEqual(render.call_args_list[0].args[0][-2:], ["--view", "full"])
            self.assertEqual(render.call_args_list[1].args[0][-2:], ["--view", "side"])
            with self.assertRaisesRegex(ValueError, "test split"):
                main(["visualize", "--model", "diffts", "--split", "val"])


if __name__ == "__main__":
    unittest.main()
