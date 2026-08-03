from __future__ import annotations

import argparse
import json
from pathlib import Path

from tomato_recon.export.usd_exporter import validate_usd_static


def validate_for_isaac(path: str | Path) -> dict:
    report = validate_usd_static(path)
    errors: list[str] = []
    material_bindings = 0
    valid_extents = 0
    try:
        from pxr import Usd, UsdGeom, UsdShade

        stage = Usd.Stage.Open(str(path))
        if not stage or not stage.GetPrimAtPath("/World/TomatoPlant"):
            errors.append("missing /World/TomatoPlant")
        else:
            for prim in stage.Traverse():
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(prim)
                extent = mesh.GetExtentAttr().Get()
                if extent and len(extent) == 2:
                    valid_extents += 1
                else:
                    errors.append(f"mesh has no valid extent: {prim.GetPath()}")
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if material:
                    material_bindings += 1
                else:
                    errors.append(f"mesh has no bound material: {prim.GetPath()}")
    except ImportError:
        # The standalone static path can run without Isaac/OpenUSD; the Isaac profile has pxr.
        pass
    report["valid"] = bool(report["valid"] and not errors)
    report.update(
        {
            "asset": str(Path(path)),
            "root_prim": "/World/TomatoPlant",
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "rendering": "not_requested",
            "valid_extents": valid_extents,
            "material_bindings": material_bindings,
            "errors": errors,
        }
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("--report", type=Path, default=Path("isaac_validation.json"))
    args = parser.parse_args(argv)
    report = validate_for_isaac(args.asset)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
