from __future__ import annotations

from torch import Tensor
import torch


def geometry_metrics(
    model_points: Tensor,
    scan_points: Tensor,
    *,
    model_normals: Tensor | None = None,
    scan_normals: Tensor | None = None,
) -> dict[str, float]:
    """Measure geometry without mixing it into the primary biological metrics.

    Normal consistency is orientation-agnostic because reconstructed and scanned
    surfaces can use opposite, but otherwise valid, normal conventions.
    """
    if not len(model_points) or not len(scan_points):
        raise ValueError("geometry metrics require non-empty point sets")
    distance = torch.cdist(model_points, scan_points)
    model_nearest, model_match = distance.min(dim=1)
    scan_nearest, scan_match = distance.min(dim=0)
    result = {
        "bidirectional_chamfer_m2": float(model_nearest.square().mean() + scan_nearest.square().mean()),
        "visible_surface_coverage_5mm": float((scan_nearest <= 0.005).float().mean()),
        "unsupported_model_fraction_10mm": float((model_nearest > 0.01).float().mean()),
    }
    if model_normals is not None and scan_normals is not None:
        if model_normals.shape != model_points.shape or scan_normals.shape != scan_points.shape:
            raise ValueError("normal arrays must match their point arrays")
        model_valid = model_normals.norm(dim=-1) > 1e-6
        scan_valid = scan_normals.norm(dim=-1) > 1e-6
        forward_valid = model_valid & scan_valid[model_match]
        reverse_valid = scan_valid & model_valid[scan_match]
        model_unit = torch.nn.functional.normalize(model_normals, dim=-1)
        scan_unit = torch.nn.functional.normalize(scan_normals, dim=-1)
        agreements = []
        if forward_valid.any():
            agreements.append(
                (model_unit[forward_valid] * scan_unit[model_match[forward_valid]]).sum(-1).abs()
            )
        if reverse_valid.any():
            agreements.append(
                (scan_unit[reverse_valid] * model_unit[scan_match[reverse_valid]]).sum(-1).abs()
            )
        result["normal_consistency"] = (
            float(torch.cat(agreements).mean()) if agreements else float("nan")
        )
    return result
