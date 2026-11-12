from __future__ import annotations

import torch

from tomato_recon.data.schemas import ParametricPlant, PlantGeometry
from tomato_recon.geometry.fruits import fruit_ellipsoid
from tomato_recon.geometry.leaves import leaf_surface
from tomato_recon.geometry.tubes import stem_tube


def generate_plant_geometry(
    plant: ParametricPlant,
    *,
    curve_samples: int = 16,
    radial_segments: int = 10,
) -> PlantGeometry:
    meshes = []
    skeleton_vertices = []
    skeleton_edges = []
    vertex_offset = 0
    for organ in plant.organs:
        mesh_count_before = len(meshes)
        controls = organ.spline_control_points
        if controls is not None:
            skeleton_vertices.append(controls)
            if len(controls) > 1:
                skeleton_edges.append(
                    torch.stack(
                        [
                            torch.arange(vertex_offset, vertex_offset + len(controls) - 1, device=controls.device),
                            torch.arange(vertex_offset + 1, vertex_offset + len(controls), device=controls.device),
                        ],
                        dim=-1,
                    )
                )
            vertex_offset += len(controls)
        if organ.organ_type in {"main_stem", "side_stem", "unknown"} and controls is not None:
            meshes.append(
                stem_tube(
                    controls,
                    organ.radius_start_m if organ.radius_start_m is not None else 0.003,
                    organ.radius_end_m if organ.radius_end_m is not None else 0.002,
                    curve_samples=curve_samples,
                    radial_segments=radial_segments,
                    name=f"stem_{organ.organ_id:03d}",
                    organ_id=organ.organ_id,
                    organ_type=organ.organ_type,
                )
            )
        elif organ.organ_type == "leaf_structure" and controls is not None:
            meshes.append(
                leaf_surface(
                    controls,
                    organ.leaf_width_coeffs
                    if organ.leaf_width_coeffs is not None
                    else [0.0, 0.01, 0.0],
                    organ.bend_coeffs,
                    curve_samples=curve_samples,
                    name=f"leaf_{organ.organ_id:03d}",
                    organ_id=organ.organ_id,
                )
            )
        elif organ.organ_type == "fruit_optional" and organ.fruit_radii_m is not None:
            if organ.confidence < 0.7:
                continue
            centre = organ.attachment_transform[:3, 3]
            meshes.append(
                fruit_ellipsoid(
                    centre,
                    organ.fruit_radii_m,
                    name=f"fruit_{organ.organ_id:03d}",
                    organ_id=organ.organ_id,
                )
            )
        if len(meshes) > mesh_count_before:
            meshes[-1].metadata.update(
                {
                    "confidence": organ.confidence,
                    "visibility": organ.visibility,
                    "parent_organ_id": organ.parent_organ_id,
                    "source_node_ids": organ.source_node_ids,
                    "parameter_json_reference": "organ_parameters.json",
                }
            )
    if not meshes:
        raise ValueError("parametric plant produced no confidence-supported geometry")
    device = meshes[0].vertices.device
    geometry = PlantGeometry(
        meshes=meshes,
        skeleton_vertices=torch.cat(skeleton_vertices) if skeleton_vertices else None,
        skeleton_edges=torch.cat(skeleton_edges) if skeleton_edges else torch.empty((0, 2), device=device, dtype=torch.long),
    )
    geometry.validate()
    visibility_scale = {"observed": 1.0, "partial": 0.6, "inferred_unknown": 0.25}
    geometry.visibility_weights = torch.cat(
        [
            torch.full(
                (len(mesh.vertices),),
                max(0.01, float(mesh.metadata.get("confidence", 1.0)))
                * visibility_scale.get(str(mesh.metadata.get("visibility", "inferred_unknown")), 0.25),
                device=mesh.vertices.device,
            )
            for mesh in meshes
        ]
    )
    return geometry
