from __future__ import annotations

import numpy as np

from .utils import double_factorial, evaluate_polynomial_with_horner


def compute_polynomial_second_moment(polynomial_vector: np.ndarray, factor_std: np.ndarray) -> np.ndarray:
    """Compute E[p(X_t)^2] when X_t is centered Gaussian with std given by factor_std."""
    coefficients = np.asarray(polynomial_vector, dtype=float)
    std_array = np.asarray(factor_std, dtype=float)

    cauchy_product = np.convolve(coefficients, coefficients)
    even_powers = np.arange(0, 2 * len(coefficients), 2)
    gaussian_moments = np.array([double_factorial(power - 1) for power in even_powers], dtype=float)
    return np.sum(
        cauchy_product[even_powers][:, None]
        * std_array[None, :] ** even_powers[:, None]
        * gaussian_moments[:, None],
        axis=0,
    )


def evaluate_polynomial_on_factor_paths(polynomial_vector: np.ndarray, factor_paths: np.ndarray) -> np.ndarray:
    """Evaluate p(X_t) on every simulated path."""
    return evaluate_polynomial_with_horner(np.asarray(polynomial_vector, dtype=float)[::-1], factor_paths)


def build_volatility_paths_from_factor_paths(
    polynomial_vector: np.ndarray,
    factor_paths: np.ndarray,
    factor_std: np.ndarray,
    forward_variance_curve: np.ndarray,
) -> np.ndarray:
    """Build sigma_t = sqrt(xi_0(t)) p(X_t) / sqrt(E[p(X_t)^2])."""
    polynomial_values = evaluate_polynomial_on_factor_paths(polynomial_vector, factor_paths)
    normalization = compute_polynomial_second_moment(polynomial_vector, factor_std)
    return np.sqrt(forward_variance_curve)[:, None] * polynomial_values / np.sqrt(normalization)[:, None]
