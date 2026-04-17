from __future__ import annotations

import numpy as np
from scipy import interpolate

from .types import (
    ForwardCurveParameters,
    ParametricForwardVarianceCurve,
    SplineForwardVarianceCurve,
    TimeDependentHModelParameters,
)


def evaluate_spline_forward_variance_curve(
    query_times: np.ndarray,
    curve: SplineForwardVarianceCurve,
    order: int = 3,
) -> np.ndarray:
    """Evaluate a positive spline by interpolating sqrt(xi_0)."""
    query_arr = np.asarray(query_times, dtype=float)
    node_times = np.asarray(curve.node_times, dtype=float)
    node_values = np.asarray(curve.node_values, dtype=float)
    safe_order = min(order, len(node_times) - 1)
    spline = interpolate.splrep(node_times, np.sqrt(np.maximum(node_values, 1e-14)), k=safe_order)
    return np.square(interpolate.splev(query_arr, spline, der=0))


def evaluate_parametric_forward_variance_curve(
    query_times: np.ndarray,
    curve: ParametricForwardVarianceCurve,
) -> np.ndarray:
    """Evaluate xi_0(t) = a exp(-b t) + c (1 - exp(-b t))."""
    query_arr = np.asarray(query_times, dtype=float)
    return curve.a * np.exp(-curve.b * query_arr) + curve.c * (1.0 - np.exp(-curve.b * query_arr))


def evaluate_forward_variance_curve(query_times: np.ndarray, curve: ForwardCurveParameters) -> np.ndarray:
    """Dispatch forward variance curve evaluation by curve type."""
    if isinstance(curve, SplineForwardVarianceCurve):
        return evaluate_spline_forward_variance_curve(query_times, curve)
    return evaluate_parametric_forward_variance_curve(query_times, curve)


def evaluate_time_dependent_h_values(
    query_times: np.ndarray,
    parameters: TimeDependentHModelParameters,
) -> np.ndarray:
    """Evaluate the published exponential interpolation for H(t)."""
    query_arr = np.asarray(query_times, dtype=float)
    if abs(parameters.h_kappa) < 1e-12:
        return np.full_like(query_arr, fill_value=parameters.h0, dtype=float)
    exp_term = np.exp(-parameters.h_kappa * query_arr)
    return parameters.h0 * exp_term + parameters.h_inf * (1.0 - exp_term)


def build_spline_forward_curve_from_anchor_values(
    anchor_times: np.ndarray,
    anchor_values: np.ndarray,
) -> SplineForwardVarianceCurve:
    """Create a spline curve object from a sampled anchor grid."""
    return SplineForwardVarianceCurve(
        node_times=np.asarray(anchor_times, dtype=float),
        node_values=np.asarray(anchor_values, dtype=float),
    )


def sample_forward_curve_on_anchor_grid(
    curve: ForwardCurveParameters,
    anchor_times: np.ndarray,
) -> np.ndarray:
    """Evaluate any supported forward curve on a fixed feature grid."""
    return evaluate_forward_variance_curve(np.asarray(anchor_times, dtype=float), curve)
