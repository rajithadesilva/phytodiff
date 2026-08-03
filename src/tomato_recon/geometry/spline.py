from __future__ import annotations

import torch
from torch import Tensor


def _clamped_knots(count: int, degree: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    interior_count = count - degree - 1
    interior = (
        torch.linspace(0, 1, interior_count + 2, device=device, dtype=dtype)[1:-1]
        if interior_count > 0
        else torch.empty(0, device=device, dtype=dtype)
    )
    return torch.cat(
        [
            torch.zeros(degree + 1, device=device, dtype=dtype),
            interior,
            torch.ones(degree + 1, device=device, dtype=dtype),
        ]
    )


def sample_cubic_bspline(control_points: Tensor, samples: int = 32) -> Tensor:
    if control_points.ndim != 2 or control_points.shape[-1] != 3:
        raise ValueError("control_points must have shape [C, 3]")
    if len(control_points) < 2 or samples < 2:
        raise ValueError("a spline needs at least two control points and two samples")
    count = len(control_points)
    degree = min(3, count - 1)
    knots = _clamped_knots(count, degree, control_points.device, control_points.dtype)
    t = torch.linspace(0, 1, samples, device=control_points.device, dtype=control_points.dtype)
    basis = torch.zeros((samples, count), device=control_points.device, dtype=control_points.dtype)
    for index in range(count):
        basis[:, index] = ((t >= knots[index]) & (t < knots[index + 1])).to(control_points.dtype)
    basis[-1] = 0
    basis[-1, -1] = 1
    for order in range(1, degree + 1):
        updated = torch.zeros_like(basis)
        for index in range(count):
            left_denominator = knots[index + order] - knots[index]
            if float(left_denominator) > 0:
                updated[:, index] += (t - knots[index]) / left_denominator * basis[:, index]
            if index + 1 < count:
                right_denominator = knots[index + order + 1] - knots[index + 1]
                if float(right_denominator) > 0:
                    updated[:, index] += (
                        (knots[index + order + 1] - t) / right_denominator * basis[:, index + 1]
                    )
        basis = updated
        basis[-1] = 0
        basis[-1, -1] = 1
    curve = basis @ control_points
    curve[0] = control_points[0]
    curve[-1] = control_points[-1]
    return curve


def arc_length(points: Tensor) -> Tensor:
    return torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).sum()

