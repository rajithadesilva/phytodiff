from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tomato_recon.data.schemas import ParametricPlant, PlantGeometry, PlantGraph, USDExportReport
from tomato_recon.export.usd_schema import (
    COLLISION_PATH,
    CONTEXT_PATH,
    GEOMETRY_GROUPS,
    GEOMETRY_PATH,
    GRAPH_PATH,
    MATERIALS_PATH,
    METADATA_PATH,
    PLANT_PATH,
    SKELETON_PATH,
    WORLD_PATH,
    valid_prim_name,
)
from tomato_recon.models.parametric.primitives import generate_plant_geometry


MATERIAL_COLOURS = {
    "main_stem": (0.08, 0.35, 0.08),
    "side_stem": (0.12, 0.45, 0.10),
    "leaf_structure": (0.04, 0.50, 0.08),
    "fruit_optional": (0.85, 0.12, 0.05),
    "unknown": (0.2, 0.4, 0.15),
}


class USDPlantExporter:
    def __init__(self, *, meters_per_unit: float = 1.0, up_axis: str = "Z") -> None:
        if meters_per_unit != 1.0 or up_axis != "Z":
            raise ValueError("USDPlantExporter supports only metres (1.0) and Z-up")
        self.meters_per_unit = meters_per_unit
        self.up_axis = up_axis

    def export(
        self,
        graph: PlantGraph,
        geometry: PlantGeometry,
        output_path: Path,
        *,
        write_debug_usda: bool,
        include_skeleton: bool,
        include_collisions: bool,
        include_context_pole: bool,
    ) -> USDExportReport:
        graph.validate()
        geometry.validate()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() != ".usd":
            raise ValueError("routine USD output_path must use the .usd extension")
        debug_path = output_path.with_name(output_path.stem + "_debug.usda") if write_debug_usda else None
        warnings = []
        try:
            self._export_with_pxr(
                graph,
                geometry,
                output_path,
                debug_path,
                include_skeleton,
                include_collisions,
                include_context_pole,
            )
            used_pxr = True
        except ImportError:
            self._export_ascii(
                graph,
                geometry,
                output_path,
                debug_path,
                include_skeleton,
                include_collisions,
                include_context_pole,
            )
            used_pxr = False
            warnings.append(
                "pxr is unavailable; .usd contains valid USDA text instead of binary crate. "
                "Install the usd extra or use the usd-export container for binary output."
            )
        report = validate_usd_static(output_path)
        warnings.extend(report.get("warnings", []))
        return USDExportReport(
            output_path=output_path,
            debug_path=debug_path,
            valid=bool(report["valid"]),
            mesh_count=int(report["mesh_count"]),
            warnings=warnings,
            used_pxr=used_pxr,
        )

    def _export_with_pxr(
        self,
        graph: PlantGraph,
        geometry: PlantGeometry,
        output_path: Path,
        debug_path: Path | None,
        include_skeleton: bool,
        include_collisions: bool,
        include_context_pole: bool,
    ) -> None:
        try:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
        except ImportError:
            raise
        stage = Usd.Stage.CreateNew(str(output_path))
        UsdGeom.SetStageMetersPerUnit(stage, self.meters_per_unit)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, WORLD_PATH).GetPrim()
        plant = UsdGeom.Xform.Define(stage, PLANT_PATH).GetPrim()
        plant.SetCustomDataByKey("schema_version", graph.schema_version)
        for path in (GEOMETRY_PATH, GRAPH_PATH, MATERIALS_PATH, METADATA_PATH):
            UsdGeom.Scope.Define(stage, path)
        if include_collisions:
            UsdGeom.Scope.Define(stage, COLLISION_PATH)
        if include_context_pole:
            UsdGeom.Scope.Define(stage, CONTEXT_PATH)
        materials = {}
        for organ_type, colour in MATERIAL_COLOURS.items():
            name = valid_prim_name(organ_type)
            material = UsdShade.Material.Define(stage, f"{MATERIALS_PATH}/{name}")
            shader = UsdShade.Shader.Define(stage, f"{MATERIALS_PATH}/{name}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            materials[organ_type] = material
        for group in set(GEOMETRY_GROUPS.values()):
            UsdGeom.Scope.Define(stage, f"{GEOMETRY_PATH}/{group}")
        for mesh_data in geometry.meshes:
            group = GEOMETRY_GROUPS.get(mesh_data.organ_type, "SideStems")
            path = f"{GEOMETRY_PATH}/{group}/{valid_prim_name(mesh_data.name)}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            vertices = mesh_data.vertices.detach().cpu().tolist()
            faces = mesh_data.faces.detach().cpu().tolist()
            mesh.CreatePointsAttr(vertices)
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
            if mesh_data.normals is not None:
                mesh.CreateNormalsAttr(mesh_data.normals.detach().cpu().tolist())
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            bounds_min = mesh_data.vertices.amin(dim=0).detach().cpu().tolist()
            bounds_max = mesh_data.vertices.amax(dim=0).detach().cpu().tolist()
            mesh.CreateExtentAttr([Gf.Vec3f(*bounds_min), Gf.Vec3f(*bounds_max)])
            prim = mesh.GetPrim()
            prim.CreateAttribute("tomato:organ_id", Sdf.ValueTypeNames.Int).Set(mesh_data.organ_id)
            prim.CreateAttribute("tomato:organ_type", Sdf.ValueTypeNames.String).Set(mesh_data.organ_type)
            prim.CreateAttribute("tomato:confidence", Sdf.ValueTypeNames.Float).Set(
                float(mesh_data.metadata.get("confidence", 1.0))
            )
            prim.CreateAttribute("tomato:visibility", Sdf.ValueTypeNames.String).Set(
                str(mesh_data.metadata.get("visibility", "inferred_unknown"))
            )
            prim.CreateAttribute("tomato:parent_organ_id", Sdf.ValueTypeNames.Int).Set(
                int(mesh_data.metadata["parent_organ_id"])
                if mesh_data.metadata.get("parent_organ_id") is not None
                else -1
            )
            prim.CreateAttribute("tomato:source_node_ids", Sdf.ValueTypeNames.IntArray).Set(
                [int(value) for value in mesh_data.metadata.get("source_node_ids", [])]
            )
            prim.CreateAttribute("tomato:parameter_json_reference", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(str(mesh_data.metadata.get("parameter_json_reference", "organ_parameters.json")))
            )
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials.get(mesh_data.organ_type, materials["unknown"]))
        if include_skeleton and graph.edges:
            curve = UsdGeom.BasisCurves.Define(stage, SKELETON_PATH)
            node = {item.id: item for item in graph.nodes}
            points = [coordinate for edge in graph.edges for coordinate in (node[edge.parent].xyz, node[edge.child].xyz)]
            curve.CreatePointsAttr(points)
            curve.CreateCurveVertexCountsAttr([2] * len(graph.edges))
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            curve.CreateWidthsAttr([0.001] * len(points))
        graph_prim = stage.GetPrimAtPath(GRAPH_PATH)
        graph_prim.CreateAttribute("tomato:json_reference", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("plant_graph.json")
        )
        metadata = stage.GetPrimAtPath(METADATA_PATH)
        metadata.CreateAttribute("tomato:plant_id", Sdf.ValueTypeNames.String).Set(graph.plant_id)
        metadata.CreateAttribute("tomato:dataset", Sdf.ValueTypeNames.String).Set(
            str(graph.source.get("dataset", "unknown"))
        )
        metadata.CreateAttribute("tomato:preprocessing_hash", Sdf.ValueTypeNames.String).Set(
            str(graph.source.get("preprocessing_hash", ""))
        )
        metadata.CreateAttribute("tomato:schema_version", Sdf.ValueTypeNames.String).Set(graph.schema_version)
        metadata.CreateAttribute("tomato:cultivar", Sdf.ValueTypeNames.String).Set(
            str(graph.source.get("cultivar", ""))
        )
        metadata.CreateAttribute("tomato:checkpoint_hashes", Sdf.ValueTypeNames.String).Set(
            json.dumps(graph.source.get("checkpoint_hashes", {}), sort_keys=True)
        )
        metadata.CreateAttribute("tomato:git_commit", Sdf.ValueTypeNames.String).Set(
            str(graph.source.get("git_commit", ""))
        )
        metadata.CreateAttribute("tomato:coordinate_transform", Sdf.ValueTypeNames.String).Set(
            json.dumps(graph.source.get("normalised_to_original", []))
        )
        stage.SetDefaultPrim(world)
        stage.GetRootLayer().Save()
        if debug_path is not None:
            stage.GetRootLayer().Export(str(debug_path))

    @staticmethod
    def _tuple(values: list[float]) -> str:
        return "(" + ", ".join(f"{float(value):.9g}" for value in values) + ")"

    def _export_ascii(
        self,
        graph: PlantGraph,
        geometry: PlantGeometry,
        output_path: Path,
        debug_path: Path | None,
        include_skeleton: bool,
        include_collisions: bool,
        include_context_pole: bool,
    ) -> None:
        lines = [
            "#usda 1.0",
            "(",
            '    defaultPrim = "World"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            ")",
            "",
            'def Xform "World"',
            "{",
            '    def Xform "TomatoPlant"',
            "    {",
            '        custom string tomato:schema_version = "1.0"',
            '        def Scope "Geometry"',
            "        {",
        ]
        groups: dict[str, list] = {value: [] for value in set(GEOMETRY_GROUPS.values())}
        for mesh in geometry.meshes:
            groups[GEOMETRY_GROUPS.get(mesh.organ_type, "SideStems")].append(mesh)
        for group, meshes in sorted(groups.items()):
            lines.extend([f'            def Scope "{group}"', "            {"])
            for mesh in meshes:
                points = ", ".join(self._tuple(v) for v in mesh.vertices.detach().cpu().tolist())
                faces = mesh.faces.detach().cpu().tolist()
                indices = ", ".join(str(int(i)) for face in faces for i in face)
                counts = ", ".join("3" for _ in faces)
                lines.extend(
                    [
                        f'                def Mesh "{valid_prim_name(mesh.name)}"',
                        "                {",
                        f"                    point3f[] points = [{points}]",
                        f"                    int[] faceVertexCounts = [{counts}]",
                        f"                    int[] faceVertexIndices = [{indices}]",
                        f"                    custom int tomato:organ_id = {mesh.organ_id}",
                        f'                    custom string tomato:organ_type = {json.dumps(mesh.organ_type)}',
                        f"                    custom int tomato:parent_organ_id = {int(mesh.metadata['parent_organ_id']) if mesh.metadata.get('parent_organ_id') is not None else -1}",
                        f"                    custom float tomato:confidence = {float(mesh.metadata.get('confidence', 1.0)):.9g}",
                        f'                    custom string tomato:visibility = {json.dumps(str(mesh.metadata.get("visibility", "inferred_unknown")))}',
                        f"                    custom int[] tomato:source_node_ids = [{', '.join(str(int(value)) for value in mesh.metadata.get('source_node_ids', []))}]",
                        "                    custom asset tomato:parameter_json_reference = @organ_parameters.json@",
                        "                }",
                    ]
                )
            lines.append("            }")
        lines.extend(["        }"])
        if include_skeleton and graph.edges:
            node = {item.id: item for item in graph.nodes}
            points = [coordinate for edge in graph.edges for coordinate in (node[edge.parent].xyz, node[edge.child].xyz)]
            lines.extend(
                [
                    '        def BasisCurves "Skeleton"',
                    "        {",
                    '            uniform token type = "linear"',
                    f"            int[] curveVertexCounts = [{', '.join('2' for _ in graph.edges)}]",
                    f"            point3f[] points = [{', '.join(self._tuple(v) for v in points)}]",
                    "        }",
                ]
            )
        lines.extend(
            [
                '        def Scope "Graph" { custom asset tomato:json_reference = @plant_graph.json@ }',
                '        def Scope "Materials"',
                "        {",
                '            def Material "main_stem" {}',
                '            def Material "side_stem" {}',
                '            def Material "leaf_structure" {}',
                '            def Material "fruit_optional" {}',
                "        }",
            ]
        )
        if include_collisions:
            lines.append('        def Scope "Collision" {}')
        lines.extend(
            [
                '        def Scope "Metadata"',
                "        {",
                f'            custom string tomato:plant_id = {json.dumps(graph.plant_id)}',
                f'            custom string tomato:dataset = {json.dumps(str(graph.source.get("dataset", "unknown")))}',
                f'            custom string tomato:preprocessing_hash = {json.dumps(str(graph.source.get("preprocessing_hash", "")))}',
                f'            custom string tomato:schema_version = {json.dumps(graph.schema_version)}',
                f'            custom string tomato:cultivar = {json.dumps(str(graph.source.get("cultivar", "")))}',
                f'            custom string tomato:checkpoint_hashes = {json.dumps(json.dumps(graph.source.get("checkpoint_hashes", {}), sort_keys=True))}',
                f'            custom string tomato:git_commit = {json.dumps(str(graph.source.get("git_commit", "")))}',
                f'            custom string tomato:coordinate_transform = {json.dumps(json.dumps(graph.source.get("normalised_to_original", [])))}',
                "        }",
                "    }",
            ]
        )
        if include_context_pole:
            lines.append('    def Scope "Context" { def Xform "SupportPole" {} }')
        lines.extend(["}", ""])
        output_path.write_text("\n".join(lines), encoding="utf-8")
        if debug_path is not None:
            shutil.copyfile(output_path, debug_path)


def validate_usd_static(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    warnings = []
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(path))
        valid = bool(stage) and bool(stage.GetPrimAtPath(PLANT_PATH))
        valid = valid and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        valid = valid and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        for prim in meshes:
            mesh = UsdGeom.Mesh(prim)
            valid = valid and bool(mesh.GetPointsAttr().Get()) and bool(mesh.GetFaceVertexIndicesAttr().Get())
        return {"valid": bool(valid and meshes), "mesh_count": len(meshes), "warnings": warnings}
    except ImportError:
        text = path.read_text(encoding="utf-8")
        required = ['metersPerUnit = 1', 'upAxis = "Z"', 'def Xform "TomatoPlant"', "def Mesh"]
        missing = [token for token in required if token not in text]
        if missing:
            warnings.append("missing USDA tokens: " + ", ".join(missing))
        return {"valid": not missing, "mesh_count": text.count("def Mesh"), "warnings": warnings}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a predicted tomato graph/parameters as OpenUSD")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args(argv)
    graph = PlantGraph.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    parameters = ParametricPlant.from_dict(json.loads(args.parameters.read_text(encoding="utf-8")))
    geometry = generate_plant_geometry(parameters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "plant_graph.json").write_text(
        json.dumps(graph.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output.parent / "organ_parameters.json").write_text(
        json.dumps(parameters.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = USDPlantExporter().export(
        graph,
        geometry,
        args.output,
        write_debug_usda=not args.no_debug,
        include_skeleton=True,
        include_collisions=False,
        include_context_pole=False,
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
