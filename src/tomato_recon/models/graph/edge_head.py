from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tomato_recon.data.schemas import EncoderOutput, SkeletonPrediction
from tomato_recon.models.graph.node_head import NodeClassificationHead, NodeScores


@dataclass
class GraphScores:
    nodes: NodeScores
    parent_scores: Tensor
    candidate_mask: Tensor


def candidate_parent_mask(
    xyz: Tensor, valid_mask: Tensor, neighbors: int, max_edge_length_m: float
) -> Tensor:
    distances = torch.cdist(xyz, xyz)
    pair_valid = valid_mask[:, :, None] & valid_mask[:, None, :]
    eye = torch.eye(xyz.shape[1], dtype=torch.bool, device=xyz.device)[None]
    distances = distances.masked_fill(~pair_valid | eye, torch.inf)
    k = min(neighbors, max(1, xyz.shape[1] - 1))
    # For each child (row), select nearest possible parents and transpose to [parent, child].
    nearest = distances.topk(k, dim=-1, largest=False).indices
    child_parent = torch.zeros_like(distances, dtype=torch.bool)
    child_parent.scatter_(2, nearest, True)
    child_parent &= distances <= max_edge_length_m
    return child_parent.transpose(1, 2)


class EdgeScoringHead(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        # h_parent, h_child, delta, distance, flow-alignment, two organ and role logits.
        feature_dim = hidden_dim * 2 + 3 + 1 + 1 + 5 * 2 + 4 * 2
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, skeleton: SkeletonPrediction, node_scores: NodeScores, candidate_mask: Tensor
    ) -> Tensor:
        """Score only sparse k-NN pairs, then scatter into the dense decoder contract."""
        hidden = node_scores.embedding
        scores = torch.full(
            candidate_mask.shape, -torch.inf, device=hidden.device, dtype=hidden.dtype
        )
        for batch in range(len(hidden)):
            pairs = candidate_mask[batch].nonzero(as_tuple=False)
            if not len(pairs):
                continue
            parent, child = pairs[:, 0], pairs[:, 1]
            delta = skeleton.node_xyz[batch, child] - skeleton.node_xyz[batch, parent]
            distance = delta.norm(dim=-1, keepdim=True)
            direction_to_parent = -delta / distance.clamp_min(1e-8)
            alignment = (
                skeleton.parent_flow[batch, child] * direction_to_parent
            ).sum(-1, keepdim=True)
            features = torch.cat(
                [
                    hidden[batch, parent],
                    hidden[batch, child],
                    delta,
                    distance,
                    alignment,
                    node_scores.organ_type_logits[batch, parent],
                    node_scores.organ_type_logits[batch, child],
                    node_scores.topology_role_logits[batch, parent],
                    node_scores.topology_role_logits[batch, child],
                ],
                dim=-1,
            )
            scores[batch, parent, child] = self.mlp(features).squeeze(-1)
        return scores


class BiologicalGraphModel(nn.Module):
    def __init__(
        self,
        point_feature_dim: int,
        hidden_dim: int = 128,
        knn_candidates: int = 12,
        max_edge_length_m: float = 0.08,
    ) -> None:
        super().__init__()
        self.node_head = NodeClassificationHead(point_feature_dim, hidden_dim)
        self.edge_head = EdgeScoringHead(hidden_dim)
        self.knn_candidates = knn_candidates
        self.max_edge_length_m = max_edge_length_m

    def forward(
        self,
        skeleton: SkeletonPrediction,
        encoder_output: EncoderOutput,
        point_mask: Tensor,
    ) -> GraphScores:
        candidates = candidate_parent_mask(
            skeleton.node_xyz,
            skeleton.valid_mask,
            self.knn_candidates,
            self.max_edge_length_m,
        )
        nodes = self.node_head(skeleton, encoder_output, point_mask)
        edge_scores = self.edge_head(skeleton, nodes, candidates)
        return GraphScores(nodes=nodes, parent_scores=edge_scores, candidate_mask=candidates)


def graph_training_loss(
    scores: GraphScores,
    organ_type: Tensor,
    topology_role: Tensor,
    visibility: Tensor,
    parent_index: Tensor,
    node_valid: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    valid = node_valid
    organ = torch.nn.functional.cross_entropy(scores.nodes.organ_type_logits[valid], organ_type[valid])
    role = torch.nn.functional.cross_entropy(scores.nodes.topology_role_logits[valid], topology_role[valid])
    vis = torch.nn.functional.cross_entropy(scores.nodes.visibility_logits[valid], visibility[valid])
    roots = torch.where(parent_index < 0, torch.ones_like(parent_index), torch.zeros_like(parent_index)).float()
    root = torch.nn.functional.binary_cross_entropy_with_logits(
        scores.nodes.root_logit[valid], roots[valid]
    )
    edge_losses = []
    for batch in range(len(parent_index)):
        children = (node_valid[batch] & (parent_index[batch] >= 0)).nonzero(as_tuple=False).flatten()
        for child in children:
            parent = parent_index[batch, child]
            logits = scores.parent_scores[batch, :, child].clone()
            if not torch.isfinite(logits[parent]):
                logits[parent] = 0.0
            edge_losses.append(torch.nn.functional.cross_entropy(logits[None], parent[None]))
    edge = torch.stack(edge_losses).mean() if edge_losses else scores.parent_scores.nan_to_num().sum() * 0
    total = organ + role + 0.5 * vis + root + edge
    return total, {"organ": organ, "role": role, "visibility": vis, "root": root, "edge": edge}
