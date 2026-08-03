from __future__ import annotations

import math
from typing import Mapping


def trait_metrics(
    prediction: Mapping[str, float], target: Mapping[str, float | None]
) -> dict[str, dict[str, float]]:
    result = {}
    for key, expected in target.items():
        if expected is None or not math.isfinite(float(expected)) or key not in prediction:
            continue
        actual = float(prediction[key])
        absolute = abs(actual - float(expected))
        result[key] = {
            "absolute_error": absolute,
            "relative_error": absolute / max(abs(float(expected)), 1e-12),
        }
    return result

