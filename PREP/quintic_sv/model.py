from __future__ import annotations

import numpy as np

from .curves import evaluate_time_dependent_h
from .utils import double_factorial


def normalization_variance(polynomial_coeffs: np.ndarray, std_x: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(polynomial_coeffs, dtype=float)
    std_x_arr = np.asarray(std_x, dtype=float)
    n = len(coeffs)
    cauchy = np.convolve(coeffs, coeffs)
    even_powers = np.arange(0, 2 * n, 2)
    gaussian_moments = np.array(
        [double_factorial(power - 1) for power in even_powers],
        dtype=float,
    )
    return np.sum(
        cauchy[even_powers][:, None]
        * std_x_arr[None, :] ** even_powers[:, None]
        * gaussian_moments[:, None],
        axis=0,
    )


def simulate_xt_grid_constant_h(
    H: float,
    eps: float,
    maturity: float,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_steps = normals.shape[0] - 1
    eta_tilde = eps ** (H - 0.5)
    kappa_tilde = (0.5 - H) / eps

    grid = np.linspace(0.0, maturity, n_steps + 1)
    exp_1 = np.exp(kappa_tilde * grid)
    exp_2 = np.exp(2.0 * kappa_tilde * grid)

    diff_exp_2 = np.concatenate(([0.0], np.diff(exp_2)))
    std_vec = np.sqrt(diff_exp_2 / (2.0 * kappa_tilde))[:, None]
    xt_full = eta_tilde * np.cumsum(std_vec * normals, axis=0) / exp_1[:, None]

    times = grid[:-1]
    xt = xt_full[:-1]
    std_x = np.sqrt(eta_tilde**2 / (2.0 * kappa_tilde) * (1.0 - np.exp(-2.0 * kappa_tilde * times)))
    return times, xt, std_x


def time_dependent_exponent(
    t: np.ndarray,
    h0: float,
    h_inf: float,
    h_kappa: float,
    eps: float,
) -> np.ndarray:
    exp_term = 1.0 - np.exp(-h_kappa * t)
    return 0.5 * t / eps - (h0 / h_kappa * exp_term + h_inf * t - h_inf / h_kappa * exp_term) / eps


def simulate_xt_grid_time_dependent_h(
    h0: float,
    h_inf: float,
    h_kappa: float,
    eps: float,
    maturity: float,
    normals: np.ndarray,
    quadrature_degree: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_steps = normals.shape[0] - 1
    grid = np.linspace(0.0, maturity, n_steps + 1)

    x_quad, w_quad = np.polynomial.legendre.leggauss(quadrature_degree)
    exp_1 = np.exp(time_dependent_exponent(grid, h0, h_inf, h_kappa, eps))

    local_nodes = 0.5 * (x_quad[None, :] + 1.0) * (grid[1:] - grid[:-1])[:, None] + grid[:-1, None]
    local_weights = 0.5 * w_quad[None, :] * (grid[1:] - grid[:-1])[:, None]

    local_exp = time_dependent_exponent(local_nodes, h0, h_inf, h_kappa, eps)
    local_h = evaluate_time_dependent_h(local_nodes, h0, h_inf, h_kappa)
    local_eta = eps ** (local_h - 0.5)

    variance_increments = np.sum(local_eta**2 * np.exp(2.0 * local_exp) * local_weights, axis=1)
    variance_increments = np.concatenate(([0.0], variance_increments))
    std_vec = np.sqrt(variance_increments)[:, None]

    std_x_full = np.sqrt(np.cumsum(variance_increments)) / exp_1
    xt_full = np.cumsum(std_vec * normals, axis=0) / exp_1[:, None]

    times = grid[:-1]
    xt = xt_full[:-1]
    std_x = std_x_full[:-1]
    return times, xt, std_x
