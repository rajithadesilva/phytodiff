from __future__ import annotations


def transition_allowed(parent_type: str, child_type: str) -> bool:
    if child_type == "main_stem":
        return parent_type == "main_stem"
    if child_type in {"side_stem", "leaf_structure"}:
        return parent_type in {"main_stem", "side_stem", "unknown"}
    if child_type == "fruit_optional":
        return parent_type == "side_stem"
    return parent_type != "fruit_optional"


def edge_type(parent_type: str, child_type: str) -> str:
    return "continuation" if parent_type == child_type else "attachment"

