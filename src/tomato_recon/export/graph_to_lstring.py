from __future__ import annotations

from tomato_recon.data.schemas import PlantGraph


def graph_to_lstring(graph: PlantGraph) -> str:
    """Optional adapter; the canonical graph remains unrestricted and non-binary."""
    graph.validate()
    children = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        children[edge.parent].append(edge.child)

    def visit(node: int) -> str:
        continuations = sorted(children[node])
        if not continuations:
            return "F"
        return "F" + "".join("[" + visit(child) + "]" for child in continuations)

    return visit(graph.root_node_id)

