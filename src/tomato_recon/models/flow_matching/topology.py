"""Directed maximum-weight arborescence and existing PlantGraph export."""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from tomato_recon.data.schemas import (
    GraphEdge, GraphNode, ORGAN_TYPE_NAMES, PlantGraph, TOPOLOGY_ROLE_NAMES,
)


@dataclass
class Prediction:
    node_xyz: Tensor
    parent_flow: Tensor
    parent_index: Tensor
    edges: Tensor
    source_slots: Tensor
    graph: PlantGraph

    @property
    def predicted_count(self):
        return len(self.node_xyz)


def rooted_tree(logits: Tensor, root: int, allowed: Tensor | None = None):
    """Chu–Liu/Edmonds via NetworkX. logits[child, parent]."""
    n = len(logits)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    scores = logits.detach().float().cpu().numpy()
    valid = np.isfinite(scores)
    valid[root, :] = False
    np.fill_diagonal(valid, False)
    if allowed is not None:
        valid &= allowed.detach().cpu().numpy()
    children, candidate_parents = np.nonzero(valid)
    graph.add_weighted_edges_from(
        (int(parent), int(child), float(scores[child, parent]))
        for child, parent in zip(children, candidate_parents)
    )
    tree = nx.maximum_spanning_arborescence(graph)
    # Build the small tree on the CPU and transfer it once, rather than launching
    # a GPU operation for each parent assignment.
    parents = torch.full((n,), -1, dtype=torch.long)
    edges = torch.tensor(sorted(tree.edges()), dtype=torch.long).reshape(-1, 2)
    parents[edges[:, 1]] = edges[:, 0]
    return parents.to(logits.device), edges.to(logits.device)


def decode(state, output, seeds, plant_id, cfg) -> Prediction:
    confidence = state[:, 3].sigmoid()
    active = confidence >= float(cfg.existence_threshold)
    if int(active.sum()) < int(cfg.min_nodes):
        active[confidence.topk(min(int(cfg.min_nodes), len(state))).indices] = True
    slots = active.nonzero().flatten()
    xyz = state[slots, :3]
    logits = output["parent_logits"][slots][:, slots]
    role = output["role_logits"][slots].argmax(-1)
    organ = output["organ_logits"][slots].argmax(-1)
    root = int(output["role_logits"][slots].softmax(-1)[:, 0].argmax())
    allowed = None
    if cfg.max_edge_length_m is not None:
        allowed = torch.cdist(xyz, xyz) <= float(cfg.max_edge_length_m)
    try:
        parents, edges = rooted_tree(logits, root, allowed)
        relaxed = False
    except nx.NetworkXException:
        # A hard distance constraint may make a connected tree impossible.
        parents, edges = rooted_tree(logits, root)
        relaxed = True
    flow = torch.zeros_like(xyz)
    valid = parents >= 0
    flow[valid] = torch.nn.functional.normalize(xyz[parents[valid]] - xyz[valid], dim=-1)
    # Export Python metadata in bulk; scalar reads inside these loops would
    # otherwise synchronize CUDA once for almost every node/edge field.
    node_values = torch.cat((xyz, confidence[slots, None]), -1).detach().cpu().tolist()
    node_labels = torch.stack((organ, role, seeds.observed[slots].long(), slots), -1).cpu().tolist()
    nodes = []
    for i, (values, labels) in enumerate(zip(node_values, node_labels)):
        organ_id, role_id, observed, source_slot = labels
        nodes.append(GraphNode(i, values[:3], ORGAN_TYPE_NAMES[organ_id],
                               TOPOLOGY_ROLE_NAMES[0 if i == root else max(1, role_id)],
                               values[3], "observed" if observed else "inferred_unknown",
                               source_slot=source_slot))
    probabilities = logits.softmax(-1)
    edge_pairs = edges.cpu().tolist()
    edge_confidence = probabilities[edges[:, 1], edges[:, 0]].detach().cpu().tolist()
    graph = PlantGraph(plant_id, root, nodes,
                       [GraphEdge(p, c, "continuation" if node_labels[p][0] == node_labels[c][0]
                                  else "attachment", confidence)
                        for (p, c), confidence in zip(edge_pairs, edge_confidence)],
                       source={"stage2_model": "aggfm", "visibility_basis": "seed_provenance"},
                       postprocessing=[{"edge_length_constraint_relaxed": relaxed}])
    graph.validate()
    return Prediction(xyz, flow, parents, edges, slots, graph)
