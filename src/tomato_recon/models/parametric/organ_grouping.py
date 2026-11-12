from __future__ import annotations

from tomato_recon.data.schemas import PlantGraph


def group_graph_organs(graph: PlantGraph) -> tuple[list[list[int]], dict[int, int]]:
    graph.validate()
    node_by_id = {node.id: node for node in graph.nodes}
    children = {node.id: [] for node in graph.nodes}
    parent = {}
    for edge in graph.edges:
        children[edge.parent].append(edge.child)
        parent[edge.child] = edge.parent
    chains = []
    visited: set[int] = set()
    for node in graph.nodes:
        parent_id = parent.get(node.id)
        starts = (
            node.id == graph.root_node_id
            or parent_id is None
            or node_by_id[parent_id].organ_type != node.organ_type
            or len(children[parent_id]) != 1
        )
        if not starts or node.id in visited:
            continue
        chain = [node.id]
        visited.add(node.id)
        while len(children[chain[-1]]) == 1:
            child = children[chain[-1]][0]
            if child in visited or node_by_id[child].organ_type != node.organ_type:
                break
            chain.append(child)
            visited.add(child)
        chains.append(chain)
    chains.extend([[node.id] for node in graph.nodes if node.id not in visited])
    node_to_organ = {
        node_id: organ_id for organ_id, chain in enumerate(chains) for node_id in chain
    }
    return chains, node_to_organ

