from __future__ import annotations

import networkx as nx
import torch
from torch import Tensor

from tomato_recon.data.schemas import (
    ORGAN_TYPE_NAMES,
    VISIBILITY_NAMES,
    GraphEdge,
    GraphNode,
    OrganType,
    PlantGraph,
    SkeletonPrediction,
)
from tomato_recon.models.graph.constraints import edge_type, transition_allowed
from tomato_recon.models.graph.edge_head import GraphScores


def select_root(scores: GraphScores, skeleton: SkeletonPrediction, batch: int, slots: Tensor) -> int:
    xyz = skeleton.node_xyz[batch, slots]
    height_range = (xyz[:, 2].max() - xyz[:, 2].min()).clamp_min(1e-6)
    base_score = -(xyz[:, 2] - xyz[:, 2].min()) / height_range
    main_probability = scores.nodes.organ_type_logits[batch, slots].softmax(-1)[:, int(OrganType.MAIN_STEM)]
    root_score = scores.nodes.root_logit[batch, slots] + main_probability + base_score
    return int(slots[root_score.argmax()])


def decode_plant_graph(
    plant_id: str,
    skeleton: SkeletonPrediction,
    scores: GraphScores,
    *,
    batch: int = 0,
    enforce_botanical_constraints: bool = True,
    allow_fruit: bool = True,
    source: dict | None = None,
) -> PlantGraph:
    slots = skeleton.valid_mask[batch].nonzero(as_tuple=False).flatten()
    if len(slots) < 2:
        raise ValueError("graph decoding requires at least two retained skeleton nodes")
    root_slot = select_root(scores, skeleton, batch, slots)
    organ_index = scores.nodes.organ_type_logits[batch].argmax(-1)
    visibility_index = scores.nodes.visibility_logits[batch].argmax(-1)
    organ_index[root_slot] = int(OrganType.MAIN_STEM)
    report: list[dict] = []

    # Unsupported fruit evidence is omitted instead of being silently attached.
    all_slots = slots.clone()
    active = set(int(slot) for slot in slots)
    for slot in list(active):
        if int(organ_index[slot]) != int(OrganType.FRUIT_OPTIONAL):
            continue
        if not allow_fruit:
            active.remove(slot)
            report.append({"action": "omit_fruit_without_pseudo_evidence", "source_slot": slot})
            continue
        has_side_parent = any(
            int(parent) in active
            and int(organ_index[parent]) == int(OrganType.SIDE_STEM)
            and bool(scores.candidate_mask[batch, parent, slot])
            for parent in slots
        )
        if not has_side_parent:
            active.remove(slot)
            report.append({"action": "omit_unsupported_fruit", "source_slot": slot})
    if len(active) < 2:
        fallback = max(
            (int(slot) for slot in all_slots if int(slot) != root_slot),
            key=lambda slot: float(skeleton.confidence[batch, slot]),
        )
        active.add(fallback)
        organ_index[fallback] = int(OrganType.UNKNOWN)
        report.append({"action": "retain_minimum_unknown_node", "source_slot": fallback})
    slots = torch.tensor(sorted(active), device=slots.device)
    if root_slot not in active:
        raise RuntimeError("root was removed during graph post-processing")

    graph = nx.DiGraph()
    graph.add_nodes_from(int(slot) for slot in slots)
    for parent_tensor in slots:
        parent = int(parent_tensor)
        for child_tensor in slots:
            child = int(child_tensor)
            if child == root_slot or parent == child:
                continue
            if not bool(scores.candidate_mask[batch, parent, child]):
                continue
            parent_type = ORGAN_TYPE_NAMES[int(organ_index[parent])]
            child_type = ORGAN_TYPE_NAMES[int(organ_index[child])]
            if enforce_botanical_constraints and not transition_allowed(parent_type, child_type):
                continue
            weight = float(scores.parent_scores[batch, parent, child].detach().cpu())
            graph.add_edge(parent, child, weight=weight, confidence=torch.sigmoid(torch.tensor(weight)).item())

    # Strongly favour a single base-to-apex chain for all main-stem nodes.
    main_slots = [
        int(slot) for slot in slots if int(organ_index[int(slot)]) == int(OrganType.MAIN_STEM)
    ]
    if root_slot not in main_slots:
        main_slots.append(root_slot)
    main_slots = sorted(
        set(main_slots), key=lambda slot: float(skeleton.node_xyz[batch, slot, 2])
    )
    if main_slots[0] != root_slot:
        main_slots.remove(root_slot)
        main_slots.insert(0, root_slot)
    finite_weights = [data["weight"] for *_, data in graph.edges(data=True)]
    boost = (max(finite_weights) if finite_weights else 0.0) + 10.0
    for parent, child in zip(main_slots, main_slots[1:], strict=False):
        graph.add_edge(parent, child, weight=boost, confidence=0.999)

    # Low-score valid fallback edges guarantee a connected candidate arborescence.
    root_type = ORGAN_TYPE_NAMES[int(organ_index[root_slot])]
    for child_tensor in slots:
        child = int(child_tensor)
        if child == root_slot:
            continue
        child_type = ORGAN_TYPE_NAMES[int(organ_index[child])]
        if (not enforce_botanical_constraints or transition_allowed(root_type, child_type)) and not graph.has_edge(root_slot, child):
            graph.add_edge(root_slot, child, weight=-1001.0, confidence=0.005)
    for child_tensor in slots:
        child = int(child_tensor)
        if child == root_slot or graph.in_degree(child):
            continue
        child_type = ORGAN_TYPE_NAMES[int(organ_index[child])]
        candidates = [
            int(parent)
            for parent in slots
            if int(parent) != child
            and (
                not enforce_botanical_constraints
                or transition_allowed(ORGAN_TYPE_NAMES[int(organ_index[int(parent)])], child_type)
            )
        ]
        if not candidates:
            # Fruit is the only type allowed to be omitted at this point.
            raise ValueError(f"no botanically valid parent for retained node slot {child}")
        parent = min(
            candidates,
            key=lambda value: float(
                torch.linalg.vector_norm(
                    skeleton.node_xyz[batch, value] - skeleton.node_xyz[batch, child]
                )
            ),
        )
        graph.add_edge(parent, child, weight=-1000.0, confidence=0.01)
        report.append({"action": "add_connectivity_fallback", "parent_slot": parent, "child_slot": child})
    for parent in list(graph.predecessors(root_slot)):
        graph.remove_edge(parent, root_slot)
    tree = nx.maximum_spanning_arborescence(graph, attr="weight", preserve_attrs=True)
    if tree.number_of_edges() != len(slots) - 1:
        raise RuntimeError("directed arborescence optimisation did not produce a spanning tree")

    slot_to_id = {int(slot): index for index, slot in enumerate(slots.tolist())}
    nodes = []
    for slot in slots.tolist():
        node_id = slot_to_id[slot]
        node_type = ORGAN_TYPE_NAMES[int(organ_index[slot])]
        if slot == root_slot:
            role = "root"
        elif tree.out_degree(slot) == 0:
            role = "tip"
        elif tree.out_degree(slot) > 1:
            role = "junction"
        else:
            role = "continuation"
        nodes.append(
            GraphNode(
                id=node_id,
                xyz=skeleton.node_xyz[batch, slot].detach().cpu().tolist(),
                organ_type=node_type,
                topology_role=role,
                existence_confidence=float(skeleton.confidence[batch, slot].detach().cpu()),
                visibility=VISIBILITY_NAMES[int(visibility_index[slot])],
                source_slot=slot,
            )
        )
    edges = []
    for parent, child, data in tree.edges(data=True):
        parent_type = ORGAN_TYPE_NAMES[int(organ_index[parent])]
        child_type = ORGAN_TYPE_NAMES[int(organ_index[child])]
        edges.append(
            GraphEdge(
                parent=slot_to_id[parent],
                child=slot_to_id[child],
                edge_type=edge_type(parent_type, child_type),
                confidence=float(data.get("confidence", 0.5)),
            )
        )
    result = PlantGraph(
        plant_id=plant_id,
        root_node_id=slot_to_id[root_slot],
        nodes=nodes,
        edges=edges,
        source=source or {},
        postprocessing=report,
    )
    result.validate()
    return result
