"""Post-process raw classifier probabilities for miner-facing ``risk_scores``."""

from __future__ import annotations

import math
from typing import Final

_EPS: Final[float] = 1e-12


def temperature_scale_probability(p: float, temperature: float) -> float:
    """
    Apply temperature scaling on logits: sigmoid(logit(p) / T).

    T > 1 spreads mass toward 0.5 (more uncertain); T < 1 sharpens.
    T <= 0 or non-finite temperature is treated as T == 1 (no-op).
    """
    if not math.isfinite(p):
        return 0.5
    p = max(_EPS, min(1.0 - _EPS, float(p)))
    if not math.isfinite(temperature) or temperature <= 0.0:
        temperature = 1.0
    if abs(temperature - 1.0) < 1e-9:
        return float(p)
    logit = math.log(p / (1.0 - p))
    z = logit / temperature
    # numerically stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        out = 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        out = ez / (1.0 + ez)
    return float(max(0.0, min(1.0, out)))
