from __future__ import annotations

import re

WORLD_PATH = "/World"
PLANT_PATH = "/World/TomatoPlant"
GEOMETRY_PATH = f"{PLANT_PATH}/Geometry"
SKELETON_PATH = f"{PLANT_PATH}/Skeleton"
GRAPH_PATH = f"{PLANT_PATH}/Graph"
MATERIALS_PATH = f"{PLANT_PATH}/Materials"
COLLISION_PATH = f"{PLANT_PATH}/Collision"
METADATA_PATH = f"{PLANT_PATH}/Metadata"
CONTEXT_PATH = f"{WORLD_PATH}/Context"

GEOMETRY_GROUPS = {
    "main_stem": "MainStem",
    "side_stem": "SideStems",
    "leaf_structure": "Leaves",
    "fruit_optional": "Fruits",
    "unknown": "SideStems",
}


def valid_prim_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not result or not result[0].isalpha():
        result = "Prim_" + result
    return result

