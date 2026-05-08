"""Tests for probability temperature scaling used at miner inference."""

import math

from poker44.training.risk_postprocess import temperature_scale_probability


def test_temperature_one_is_identity():
    for p in (0.01, 0.3, 0.5, 0.7, 0.99):
        out = temperature_scale_probability(p, 1.0)
        assert abs(out - p) < 1e-6


def test_temperature_gt_one_pulls_toward_half():
    p = 0.95
    out = temperature_scale_probability(p, 2.0)
    assert out < p and out > 0.5


def test_temperature_invalid_falls_back():
    assert abs(temperature_scale_probability(0.7, -1.0) - 0.7) < 1e-6


def test_nan_input_returns_half():
    assert temperature_scale_probability(float("nan"), 1.0) == 0.5
