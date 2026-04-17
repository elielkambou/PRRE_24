from __future__ import annotations

import numpy as np

from .curves import evaluate_time_dependent_h_values
from .types import ConstantHModelParameters, FactorPathSimulation, ModelParameters, TimeDependentHModelParameters


def compute_constant_h_mean_reversion(model_parameters: ConstantHModelParameters) -> float:
    """Return the OU mean reversion (1/2 - H) / epsilon."""
    return (0.5 - model_parameters.h_value) / model_parameters.epsilon


def compute_constant_h_vol_of_vol_scale(model_parameters: ConstantHModelParameters) -> float:
    """Return epsilon^(H - 1/2), the diffusion scale of X_t."""
    return model_parameters.epsilon ** (model_parameters.h_value - 0.5)


def simulate_factor_paths_with_constant_h(
    model_parameters: ConstantHModelParameters,
    maturity: float,
    normals: np.ndarray,
) -> FactorPathSimulation:
    """Simulate the Gaussian OU factor when H is constant."""
    n_steps = normals.shape[0] - 1
    mean_reversion = compute_constant_h_mean_reversion(model_parameters)
    diffusion_scale = compute_constant_h_vol_of_vol_scale(model_parameters)

    grid = np.linspace(0.0, maturity, n_steps + 1)
    exp_level = np.exp(mean_reversion * grid)
    exp_twice = np.exp(2.0 * mean_reversion * grid)

    variance_increments = np.concatenate(([0.0], np.diff(exp_twice))) / (2.0 * mean_reversion)
    std_increments = np.sqrt(variance_increments)[:, None]
    full_paths = diffusion_scale * np.cumsum(std_increments * normals, axis=0) / exp_level[:, None]

    times = grid[:-1]
    factor_paths = full_paths[:-1]
    factor_std = np.sqrt(
        diffusion_scale**2
        / (2.0 * mean_reversion)
        * (1.0 - np.exp(-2.0 * mean_reversion * times))
    )
    return FactorPathSimulation(times=times, factor_paths=factor_paths, factor_std=factor_std)


def compute_time_dependent_exponent(
    query_times: np.ndarray,
    model_parameters: TimeDependentHModelParameters,
) -> np.ndarray:
    """Closed-form exponent used to simulate the time-dependent-H factor."""
    times = np.asarray(query_times, dtype=float)
    if abs(model_parameters.h_kappa) < 1e-12:
        return (0.5 - model_parameters.h0) * times / model_parameters.epsilon

    exp_term = 1.0 - np.exp(-model_parameters.h_kappa * times)
    numerator = (
        model_parameters.h0 / model_parameters.h_kappa * exp_term
        + model_parameters.h_inf * times
        - model_parameters.h_inf / model_parameters.h_kappa * exp_term
    )
    return 0.5 * times / model_parameters.epsilon - numerator / model_parameters.epsilon


def simulate_factor_paths_with_time_dependent_h(
    model_parameters: TimeDependentHModelParameters,
    maturity: float,
    normals: np.ndarray,
    inner_quadrature_degree: int = 100,
) -> FactorPathSimulation:
    """Simulate the Gaussian factor when H depends on time."""
    n_steps = normals.shape[0] - 1
    grid = np.linspace(0.0, maturity, n_steps + 1)

    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(inner_quadrature_degree)
    exp_grid = np.exp(compute_time_dependent_exponent(grid, model_parameters))

    local_nodes = 0.5 * (quadrature_nodes[None, :] + 1.0) * (grid[1:] - grid[:-1])[:, None] + grid[:-1, None]
    local_weights = 0.5 * quadrature_weights[None, :] * (grid[1:] - grid[:-1])[:, None]

    local_exponent = compute_time_dependent_exponent(local_nodes, model_parameters)
    local_h_values = evaluate_time_dependent_h_values(local_nodes, model_parameters)
    local_diffusion_scale = model_parameters.epsilon ** (local_h_values - 0.5)

    variance_increments = np.sum(local_diffusion_scale**2 * np.exp(2.0 * local_exponent) * local_weights, axis=1)
    variance_increments = np.concatenate(([0.0], variance_increments))

    std_increments = np.sqrt(variance_increments)[:, None]
    full_factor_std = np.sqrt(np.cumsum(variance_increments)) / exp_grid
    full_paths = np.cumsum(std_increments * normals, axis=0) / exp_grid[:, None]

    return FactorPathSimulation(
        times=grid[:-1],
        factor_paths=full_paths[:-1],
        factor_std=full_factor_std[:-1],
    )


def simulate_factor_paths_for_model(
    model_parameters: ModelParameters,
    maturity: float,
    normals: np.ndarray,
    inner_quadrature_degree: int = 100,
) -> FactorPathSimulation:
    """Dispatch the factor simulation based on the type of H."""
    if isinstance(model_parameters, ConstantHModelParameters):
        return simulate_factor_paths_with_constant_h(model_parameters, maturity, normals)
    return simulate_factor_paths_with_time_dependent_h(
        model_parameters,
        maturity,
        normals,
        inner_quadrature_degree=inner_quadrature_degree,
    )
