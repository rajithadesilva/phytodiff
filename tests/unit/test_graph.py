from __future__ import annotations

import unittest

import networkx as nx
import torch

from tomato_recon.data.schemas import SkeletonPrediction
from tomato_recon.data.processed import make_tiny_sample
from tomato_recon.models.graph.constraints import transition_allowed
from tomato_recon.models.graph.decode import decode_plant_graph
from tomato_recon.models.graph.edge_head import GraphScores, candidate_parent_mask
from tomato_recon.models.graph.node_head import NodeScores


class GraphTests(unittest.TestCase):
    def test_candidate_edges(self) -> None:
        xyz = torch.tensor([[[0.0, 0, 0], [0, 0, 0.01], [0, 0, 0.03]]])
        mask = candidate_parent_mask(xyz, torch.ones(1, 3, dtype=torch.bool), 1, 0.02)
        self.assertEqual(mask.shape, (1, 3, 3))
        self.assertFalse(bool(mask.diagonal(dim1=1, dim2=2).any()))

    def test_transition_masks(self) -> None:
        self.assertFalse(transition_allowed("leaf_structure", "main_stem"))
        self.assertFalse(transition_allowed("main_stem", "fruit_optional"))
        self.assertTrue(transition_allowed("side_stem", "fruit_optional"))

    def test_rooted_acyclic_arborescence(self) -> None:
        sample = make_tiny_sample(16, 64)
        valid = sample.node_valid[None]
        skeleton = SkeletonPrediction(
            node_xyz=sample.node_xyz[None],
            parent_flow=sample.parent_flow[None],
            existence_logit=torch.where(valid, 8.0, -8.0),
            confidence=valid.float(),
            valid_mask=valid,
        )
        organ_logits = torch.full((1, 16, 5), -5.0)
        role_logits = torch.full((1, 16, 4), -5.0)
        visibility_logits = torch.zeros((1, 16, 3))
        organ_logits[0, torch.arange(16), sample.organ_type] = 5.0
        role_logits[0, torch.arange(16), sample.topology_role] = 5.0
        nodes = NodeScores(
            embedding=torch.zeros(1, 16, 8),
            organ_type_logits=organ_logits,
            topology_role_logits=role_logits,
            visibility_logits=visibility_logits,
            root_logit=torch.tensor([[9.0] + [0.0] * 15]),
        )
        candidates = valid[:, :, None] & valid[:, None, :]
        candidates &= ~torch.eye(16, dtype=torch.bool)[None]
        parent_scores = torch.full((1, 16, 16), -10.0)
        for child in range(1, 9):
            parent_scores[0, sample.parent_index[child], child] = 10.0
        scores = GraphScores(nodes, parent_scores, candidates)
        graph = decode_plant_graph("fixture", skeleton, scores)
        graph.validate()
        directed = nx.DiGraph((edge.parent, edge.child) for edge in graph.edges)
        self.assertTrue(nx.is_directed_acyclic_graph(directed))
        self.assertEqual(len(graph.edges), len(graph.nodes) - 1)
        self.assertEqual(graph.nodes[graph.root_node_id].organ_type, "main_stem")


if __name__ == "__main__":
    unittest.main()
