from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from tomato_recon.data.schemas import (
    ParametricPlant,
    PlantGeometry,
    PlantGraph,
    SkeletonPrediction,
)


@dataclass
class PipelineResult:
    skeleton: SkeletonPrediction
    graph: PlantGraph
    parameters: ParametricPlant
    geometry: PlantGeometry
    traits: dict[str, Any]
    uncertainty: dict[str, Any]
    semantic_logits: torch.Tensor


def derive_traits(graph: PlantGraph, geometry: PlantGeometry) -> dict[str, Any]:
    node = {item.id: torch.tensor(item.xyz) for item in graph.nodes}
    lengths = [float(torch.linalg.vector_norm(node[e.child] - node[e.parent])) for e in graph.edges]
    z = torch.tensor([item.xyz[2] for item in graph.nodes])
    return {
        "plant_height_m": float(z.max() - z.min()),
        "total_skeleton_length_m": float(sum(lengths)),
        "mean_internode_length_m": float(sum(lengths) / max(len(lengths), 1)),
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "leaf_structure_count": sum(n.organ_type == "leaf_structure" for n in graph.nodes),
        "side_stem_count": sum(n.organ_type == "side_stem" for n in graph.nodes),
        "fruit_optional_count": sum(n.organ_type == "fruit_optional" for n in graph.nodes),
        "mesh_count": len(geometry.meshes),
    }


class TomatoReconstructionPipeline(nn.Module):
    """The full reconstruction pipeline awaits variable-length DiffTS integration."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        raise RuntimeError("Stages 3–5 await DiffTS stage 2 integration")
