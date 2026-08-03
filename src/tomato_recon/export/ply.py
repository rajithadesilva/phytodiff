from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tomato_recon.data.schemas import MeshData


def write_point_ply(
    path: str | Path,
    xyz: torch.Tensor | np.ndarray,
    rgb: torch.Tensor | np.ndarray | None = None,
    edges: torch.Tensor | np.ndarray | None = None,
) -> Path:
    path = Path(path)
    points = np.asarray(xyz.detach().cpu() if isinstance(xyz, torch.Tensor) else xyz, dtype=np.float32)
    colours = None
    if rgb is not None:
        colours = np.asarray(rgb.detach().cpu() if isinstance(rgb, torch.Tensor) else rgb)
        if colours.max(initial=0) <= 1:
            colours = colours * 255
        colours = np.clip(colours, 0, 255).astype(np.uint8)
    edge_array = None
    if edges is not None:
        edge_array = np.asarray(edges.detach().cpu() if isinstance(edges, torch.Tensor) else edges, dtype=np.int64)
    lines = ["ply", "format ascii 1.0", f"element vertex {len(points)}", "property float x", "property float y", "property float z"]
    if colours is not None:
        lines.extend(["property uchar red", "property uchar green", "property uchar blue"])
    if edge_array is not None:
        lines.extend([f"element edge {len(edge_array)}", "property int vertex1", "property int vertex2"])
    lines.append("end_header")
    for index, point in enumerate(points):
        line = " ".join(f"{float(value):.9g}" for value in point)
        if colours is not None:
            line += " " + " ".join(str(int(value)) for value in colours[index])
        lines.append(line)
    if edge_array is not None:
        lines.extend(f"{int(edge[0])} {int(edge[1])}" for edge in edge_array)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_mesh_ply(path: str | Path, mesh: MeshData) -> Path:
    path = Path(path)
    mesh.validate()
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    normals = mesh.normals.detach().cpu().numpy() if mesh.normals is not None else None
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if normals is not None:
        lines.extend(["property float nx", "property float ny", "property float nz"])
    lines.extend(
        [f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    )
    for index, vertex in enumerate(vertices):
        values = vertex if normals is None else np.concatenate([vertex, normals[index]])
        lines.append(" ".join(f"{float(value):.9g}" for value in values))
    lines.extend("3 " + " ".join(str(int(index)) for index in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
