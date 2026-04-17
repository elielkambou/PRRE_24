from __future__ import annotations

import math
from pathlib import Path

import numpy as np


TRADING_DAYS_PER_YEAR = 365.0
VIX_WINDOW_YEARS = 30.0 / 360.0


def double_factorial(n: int) -> int:
    """Return n!! with the convention (-1)!! = 0!! = 1."""
    if n <= 0:
        return 1
    return math.prod(range(n, 0, -2))


def evaluate_polynomial_with_horner(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial on an array using Horner's method."""
    coeffs = np.asarray(coefficients, dtype=float)
    result = np.zeros_like(x, dtype=float) + coeffs[0]
    for coefficient in coeffs[1:]:
        result = result * x + coefficient
    return result


def generate_antithetic_normals(n_steps: int, n_base_paths: int, seed: int) -> np.ndarray:
    """Generate Gaussian increments and append their antithetic counterparts."""
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((n_steps, n_base_paths))
    antithetic = np.concatenate((normals, -normals), axis=1)
    zeros = np.zeros((1, antithetic.shape[1]), dtype=float)
    return np.vstack((zeros, antithetic))


def scaled_int(value: int, scale: float, minimum: int) -> int:
    """Scale an integer budget while enforcing a lower bound."""
    return max(minimum, int(round(value * scale)))


def default_spx_log_moneyness_range(maturity: float) -> tuple[float, float]:
    """Heuristic strike range used by the paper figures."""
    if maturity < 2 / 52:
        return (-0.15, 0.03)
    if maturity < 1 / 12:
        return (-0.25, 0.05)
    if maturity < 2 / 12:
        return (-0.4, 0.1)
    if maturity < 3 / 12:
        return (-0.6, 0.1)
    if maturity < 6 / 12:
        return (-0.7, 0.15)
    return (-1.0, 0.2)


def maturity_to_days(maturity: float) -> int:
    """Convert a year fraction into rounded calendar days."""
    return int(round(maturity * TRADING_DAYS_PER_YEAR))


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
