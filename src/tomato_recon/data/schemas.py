"""Typed contracts shared by every pipeline stage.

Dataclasses carry tensors in memory. Cache and checkpoint writers convert them to
plain NPZ/JSON structures with explicit schema versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

SCHEMA_VERSION = "1.0"
IGNORE_INDEX = -100


class SemanticClass(IntEnum):
    BACKGROUND = 0
    LEAF = 1
    MAIN_STEM = 2
    SUPPORT_POLE = 3
    SIDE_STEM = 4


class OrganType(IntEnum):
    MAIN_STEM = 0
    SIDE_STEM = 1
    LEAF_STRUCTURE = 2
    FRUIT_OPTIONAL = 3
    UNKNOWN = 4


class TopologyRole(IntEnum):
    ROOT = 0
    CONTINUATION = 1
    JUNCTION = 2
    TIP = 3


class Visibility(IntEnum):
    OBSERVED = 0
    PARTIAL = 1
    INFERRED_UNKNOWN = 2


ORGAN_TYPE_NAMES = {
    int(OrganType.MAIN_STEM): "main_stem",
    int(OrganType.SIDE_STEM): "side_stem",
    int(OrganType.LEAF_STRUCTURE): "leaf_structure",
    int(OrganType.FRUIT_OPTIONAL): "fruit_optional",
    int(OrganType.UNKNOWN): "unknown",
}
TOPOLOGY_ROLE_NAMES = {
    int(TopologyRole.ROOT): "root",
    int(TopologyRole.CONTINUATION): "continuation",
    int(TopologyRole.JUNCTION): "junction",
    int(TopologyRole.TIP): "tip",
}
VISIBILITY_NAMES = {
    int(Visibility.OBSERVED): "observed",
    int(Visibility.PARTIAL): "partial",
    int(Visibility.INFERRED_UNKNOWN): "inferred_unknown",
}


@dataclass
class FeatureMap:
    xyz: Tensor
    features: Tensor
    mask: Tensor


@dataclass
class GraphNode:
    id: int
    xyz: list[float]
    organ_type: str
    topology_role: str
    existence_confidence: float
    visibility: str
    radius_m: float | None = None
    source_slot: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": int(self.id),
            "xyz": [float(v) for v in self.xyz],
            "organ_type": self.organ_type,
            "topology_role": self.topology_role,
            "existence_confidence": float(self.existence_confidence),
            "visibility": self.visibility,
        }
        if self.radius_m is not None:
            result["radius_m"] = float(self.radius_m)
        if self.source_slot is not None:
            result["source_slot"] = int(self.source_slot)
        return result


@dataclass
class GraphEdge:
    parent: int
    child: int
    edge_type: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent": int(self.parent),
            "child": int(self.child),
            "edge_type": self.edge_type,
            "confidence": float(self.confidence),
        }


@dataclass
class PlantGraph:
    plant_id: str
    root_node_id: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    organs: list[dict[str, Any]] = field(default_factory=list)
    traits: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    coordinate_frame: dict[str, Any] = field(
        default_factory=lambda: {"up_axis": "Z", "meters_per_unit": 1.0}
    )
    postprocessing: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("plant graph node IDs must be unique")
        if self.root_node_id not in ids:
            raise ValueError("plant graph root is missing")
        parents: dict[int, int] = {}
        adjacency: dict[int, list[int]] = {i: [] for i in ids}
        for edge in self.edges:
            if edge.parent not in ids or edge.child not in ids:
                raise ValueError("edge references a missing node")
            if edge.child == self.root_node_id:
                raise ValueError("root must not have a parent")
            if edge.child in parents:
                raise ValueError("every non-root node must have exactly one parent")
            parents[edge.child] = edge.parent
            adjacency[edge.parent].append(edge.child)
        if set(parents) != ids - {self.root_node_id}:
            raise ValueError("graph is not a connected rooted tree")
        seen: set[int] = set()
        stack = [self.root_node_id]
        while stack:
            node = stack.pop()
            if node in seen:
                raise ValueError("graph contains a cycle")
            seen.add(node)
            stack.extend(adjacency[node])
        if seen != ids:
            raise ValueError("graph contains disconnected nodes")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plant_id": self.plant_id,
            "coordinate_frame": self.coordinate_frame,
            "root_node_id": int(self.root_node_id),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "organs": self.organs,
            "traits": self.traits,
            "source": self.source,
            "postprocessing": self.postprocessing,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlantGraph":
        graph = cls(
            plant_id=value["plant_id"],
            root_node_id=int(value["root_node_id"]),
            nodes=[GraphNode(**n) for n in value["nodes"]],
            edges=[GraphEdge(**e) for e in value["edges"]],
            organs=value.get("organs", []),
            traits=value.get("traits", {}),
            source=value.get("source", {}),
            coordinate_frame=value.get(
                "coordinate_frame", {"up_axis": "Z", "meters_per_unit": 1.0}
            ),
            postprocessing=value.get("postprocessing", []),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )
        graph.validate()
        return graph


@dataclass
class OrganParameters:
    organ_id: int
    organ_type: str
    parent_organ_id: int | None
    attachment_transform: Tensor
    spline_control_points: Tensor | None = None
    radius_start_m: float | Tensor | None = None
    radius_end_m: float | Tensor | None = None
    leaf_length_m: float | Tensor | None = None
    leaf_width_coeffs: list[float] | Tensor | None = None
    bend_coeffs: list[float] | Tensor | None = None
    fruit_radii_m: list[float] | Tensor | None = None
    confidence: float = 1.0
    source_node_ids: list[int] = field(default_factory=list)
    visibility: str = "observed"

    @staticmethod
    def _plain(value: Any) -> Any:
        if isinstance(value, Tensor):
            value = value.detach().cpu()
            return float(value) if value.numel() == 1 else value.tolist()
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "organ_id": int(self.organ_id),
            "organ_type": self.organ_type,
            "parent_organ_id": self.parent_organ_id,
            "attachment_transform": self._plain(self.attachment_transform),
            "spline_control_points": self._plain(self.spline_control_points),
            "radius_start_m": self._plain(self.radius_start_m),
            "radius_end_m": self._plain(self.radius_end_m),
            "leaf_length_m": self._plain(self.leaf_length_m),
            "leaf_width_coeffs": self._plain(self.leaf_width_coeffs),
            "bend_coeffs": self._plain(self.bend_coeffs),
            "fruit_radii_m": self._plain(self.fruit_radii_m),
            "confidence": float(self.confidence),
            "source_node_ids": [int(v) for v in self.source_node_ids],
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OrganParameters":
        copied = dict(value)
        copied["attachment_transform"] = torch.tensor(
            copied["attachment_transform"], dtype=torch.float32
        )
        for key in ("spline_control_points", "leaf_width_coeffs", "bend_coeffs", "fruit_radii_m"):
            if copied.get(key) is not None:
                copied[key] = torch.tensor(copied[key], dtype=torch.float32)
        return cls(**copied)


@dataclass
class ParametricPlant:
    plant_id: str
    organs: list[OrganParameters]
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plant_id": self.plant_id,
            "organs": [o.to_dict() for o in self.organs],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ParametricPlant":
        return cls(
            plant_id=value["plant_id"],
            organs=[OrganParameters.from_dict(v) for v in value.get("organs", [])],
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            metadata=value.get("metadata", {}),
        )


@dataclass
class MeshData:
    name: str
    vertices: Tensor
    faces: Tensor
    normals: Tensor | None = None
    organ_id: int = -1
    organ_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[-1] != 3:
            raise ValueError("mesh vertices must have shape [V, 3]")
        if self.faces.ndim != 2 or self.faces.shape[-1] != 3:
            raise ValueError("mesh faces must have shape [F, 3]")
        if not torch.isfinite(self.vertices).all():
            raise ValueError("mesh vertices contain NaN or infinity")
        if self.normals is not None:
            if self.normals.shape != self.vertices.shape:
                raise ValueError("mesh normals must match the vertex array")
            if not torch.isfinite(self.normals).all():
                raise ValueError("mesh normals contain NaN or infinity")
        if self.faces.numel() and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("mesh face index is out of range")


@dataclass
class PlantGeometry:
    meshes: list[MeshData]
    skeleton_vertices: Tensor | None = None
    skeleton_edges: Tensor | None = None
    visibility_weights: Tensor | None = None

    def validate(self) -> None:
        if not self.meshes:
            raise ValueError("plant geometry must contain at least one mesh")
        for mesh in self.meshes:
            mesh.validate()

    def combined_mesh(self) -> MeshData:
        self.validate()
        vertices: list[Tensor] = []
        faces: list[Tensor] = []
        normals: list[Tensor] = []
        has_all_normals = True
        offset = 0
        for mesh in self.meshes:
            vertices.append(mesh.vertices)
            faces.append(mesh.faces + offset)
            if mesh.normals is None:
                has_all_normals = False
            else:
                normals.append(mesh.normals)
            offset += len(mesh.vertices)
        return MeshData(
            name="reconstructed_plant",
            vertices=torch.cat(vertices),
            faces=torch.cat(faces) if faces else torch.empty((0, 3), dtype=torch.long),
            normals=torch.cat(normals) if has_all_normals else None,
        )


@dataclass
class PlantSample:
    plant_id: str
    xyz: Tensor
    rgb: Tensor
    normals: Tensor
    semantic: Tensor
    instance: Tensor
    point_valid: Tensor
    node_xyz: Tensor
    parent_flow: Tensor
    node_valid: Tensor
    parent_index: Tensor
    organ_type: Tensor
    topology_role: Tensor
    visibility: Tensor
    graph_target: PlantGraph | None = None
    param_target: ParametricPlant | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n, k = len(self.xyz), len(self.node_xyz)
        point_shapes = {
            "xyz": (n, 3), "rgb": (n, 3), "normals": (n, 3),
            "semantic": (n,), "instance": (n,), "point_valid": (n,),
        }
        node_shapes = {
            "node_xyz": (k, 3), "parent_flow": (k, 3), "node_valid": (k,),
            "parent_index": (k,), "organ_type": (k,), "topology_role": (k,),
            "visibility": (k,),
        }
        for name, shape in {**point_shapes, **node_shapes}.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(getattr(self, name).shape)}")
        if not torch.isfinite(self.xyz).all() or not torch.isfinite(self.node_xyz).all():
            raise ValueError("coordinates must be finite")


@dataclass
class PlantBatch:
    plant_ids: list[str]
    xyz: Tensor
    rgb: Tensor
    normals: Tensor
    semantic: Tensor
    instance: Tensor
    point_valid: Tensor
    node_xyz: Tensor
    parent_flow: Tensor
    node_valid: Tensor
    parent_index: Tensor
    organ_type: Tensor
    topology_role: Tensor
    visibility: Tensor
    samples: list[PlantSample]

    def to(self, device: torch.device | str) -> "PlantBatch":
        values = self.__dict__.copy()
        for key, value in values.items():
            if isinstance(value, Tensor):
                values[key] = value.to(device)
        return PlantBatch(**values)


@dataclass
class EncoderOutput:
    point_xyz: Tensor
    point_features: Tensor
    multiscale_features: list[FeatureMap]
    global_feature: Tensor
    semantic_logits: Tensor
    skeleton_logits: Tensor
    centreline_offset: Tensor
    junction_logits: Tensor


@dataclass
class SkeletonPrediction:
    node_xyz: Tensor
    parent_flow: Tensor
    existence_logit: Tensor
    confidence: Tensor
    valid_mask: Tensor
    sample_id: Tensor | None = None


@dataclass
class USDExportReport:
    output_path: Path
    debug_path: Path | None
    valid: bool
    mesh_count: int
    warnings: list[str] = field(default_factory=list)
    used_pxr: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "output_path": str(self.output_path),
            "debug_path": str(self.debug_path) if self.debug_path else None,
            "valid": self.valid,
            "mesh_count": self.mesh_count,
            "warnings": self.warnings,
            "used_pxr": self.used_pxr,
        }
