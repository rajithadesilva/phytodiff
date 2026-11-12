from __future__ import annotations

from tomato_recon.data.schemas import PlantGraph


def graph_metrics(prediction: PlantGraph, target: PlantGraph) -> dict[str, float | int]:
    prediction.validate()
    target.validate()
    predicted_edges = {(edge.parent, edge.child) for edge in prediction.edges}
    target_edges = {(edge.parent, edge.child) for edge in target.edges}
    intersection = len(predicted_edges & target_edges)
    predicted_type = {node.id: node.organ_type for node in prediction.nodes}
    target_type = {node.id: node.organ_type for node in target.nodes}
    predicted_stem = {
        edge
        for edge in predicted_edges
        if predicted_type[edge[0]] == "main_stem" and predicted_type[edge[1]] == "main_stem"
    }
    target_stem = {
        edge
        for edge in target_edges
        if target_type[edge[0]] == "main_stem" and target_type[edge[1]] == "main_stem"
    }
    return {
        "edge_precision": intersection / max(len(predicted_edges), 1),
        "edge_recall": intersection / max(len(target_edges), 1),
        "root_accuracy": float(prediction.root_node_id == target.root_node_id),
        "connected_components": 1,
        "cycle_rate": 0.0,
        "main_stem_path_accuracy": len(predicted_stem & target_stem)
        / max(len(predicted_stem | target_stem), 1),
        "graph_edit_proxy": float(
            len(predicted_edges ^ target_edges) + abs(len(prediction.nodes) - len(target.nodes))
        ),
        "node_count_error": abs(len(prediction.nodes) - len(target.nodes)),
    }
