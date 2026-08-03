from __future__ import annotations

import torch
from torch import Tensor


def expected_calibration_error(confidence: Tensor, correct: Tensor, bins: int = 10) -> float:
    error = torch.zeros((), device=confidence.device)
    boundaries = torch.linspace(0, 1, bins + 1, device=confidence.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (confidence >= lower) & (confidence < upper)
        if selected.any():
            error += selected.float().mean() * (
                confidence[selected].mean() - correct[selected].float().mean()
            ).abs()
    return float(error)

