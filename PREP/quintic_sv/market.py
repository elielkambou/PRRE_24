from __future__ import annotations

import numpy as np

from .curves import evaluate_spline_forward_curve
from .types import ForwardCurveComparison, SPXSmileResult, VIXSmileResult


def _simulate_bid_ask(
    x_values: np.ndarray,
    model_iv: np.ndarray,
    rng: np.random.Generator,
    relative_noise: float,
    relative_spread: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_centered = (x_values - np.mean(x_values)) / (np.ptp(x_values) + 1e-12)
    smooth_pattern = 0.65 * np.sin(2.8 * x_centered) + 0.35 * np.cos(5.2 * x_centered)
    random_pattern = rng.normal(loc=0.0, scale=0.35, size=model_iv.shape)
    mid = model_iv * (1.0 + relative_noise * (smooth_pattern + random_pattern))
    half_spread = np.maximum(0.004, relative_spread * model_iv * (1.0 + 0.2 * np.abs(x_centered)))
    bid = np.maximum(mid - half_spread, 1e-4)
    ask = mid + half_spread
    return bid, ask


def add_synthetic_market_to_spx(smile: SPXSmileResult, seed: int) -> SPXSmileResult:
    rng = np.random.default_rng(seed)
    bid, ask = _simulate_bid_ask(smile.log_moneyness, smile.model_iv, rng, 0.025, 0.06)
    smile.market_bid_iv = bid
    smile.market_ask_iv = ask
    return smile


def add_synthetic_market_to_vix(smile: VIXSmileResult, seed: int) -> VIXSmileResult:
    rng = np.random.default_rng(seed)
    bid, ask = _simulate_bid_ask(smile.strikes, smile.model_iv, rng, 0.04, 0.035)
    smile.market_bid_iv = bid
    smile.market_ask_iv = ask
    smile.market_future = smile.future * (1.0 + rng.normal(loc=0.0, scale=0.006))
    return smile


def simulate_forward_curve_comparison(
    node_times: np.ndarray,
    calibrated_nodes: np.ndarray,
    seed: int,
    grid_size: int = 2_000,
) -> ForwardCurveComparison:
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 2.5 * np.pi, len(node_times))
    perturbation = 1.0 + 0.12 * np.sin(phase) - 0.06 * np.cos(1.7 * phase) + rng.normal(0.0, 0.03, len(node_times))
    stripped_nodes = np.maximum(calibrated_nodes * perturbation, 1e-4)
    grid_times = np.linspace(0.0, float(node_times[-1]), grid_size)
    calibrated_curve = evaluate_spline_forward_curve(grid_times, node_times, calibrated_nodes)
    stripped_curve = evaluate_spline_forward_curve(grid_times, node_times, stripped_nodes)
    return ForwardCurveComparison(
        node_times=node_times,
        calibrated_nodes=calibrated_nodes,
        stripped_nodes=stripped_nodes,
        grid_times=grid_times,
        calibrated_curve=calibrated_curve,
        stripped_curve=stripped_curve,
    )
