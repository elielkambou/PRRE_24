from __future__ import annotations

import numpy as np
from scipy import interpolate


def evaluate_spline_forward_curve(
    query_times: np.ndarray,
    node_times: np.ndarray,
    node_values: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    safe_order = min(order, len(node_times) - 1)
    spline = interpolate.splrep(node_times, np.sqrt(np.maximum(node_values, 1e-14)), k=safe_order)
    return np.square(interpolate.splev(query_times, spline, der=0))


def evaluate_parametric_forward_curve(
    query_times: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    return a * np.exp(-b * query_times) + c * (1.0 - np.exp(-b * query_times))


def evaluate_time_dependent_h(
    query_times: np.ndarray,
    h0: float,
    h_inf: float,
    h_kappa: float,
) -> np.ndarray:
    return h0 * np.exp(-h_kappa * query_times) + h_inf * (1.0 - np.exp(-h_kappa * query_times))
