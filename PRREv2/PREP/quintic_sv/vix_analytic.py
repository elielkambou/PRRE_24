from __future__ import annotations

import math

import numpy as np
from scipy.special import comb

from .black import implied_volatility_vector
from .curves import evaluate_forward_variance_curve, evaluate_time_dependent_h_values
from .factor_process import (
    compute_constant_h_mean_reversion,
    compute_constant_h_vol_of_vol_scale,
    compute_time_dependent_exponent,
)
from .polynomial_volatility import compute_polynomial_second_moment
from .types import (
    ConstantHModelParameters,
    ExperimentInput,
    TimeDependentHModelParameters,
    VIXSmileResult,
)
from .utils import VIX_WINDOW_YEARS, double_factorial, evaluate_polynomial_with_horner


def gaussian_density(x_values: np.ndarray) -> np.ndarray:
    """Standard Gaussian density."""
    return np.exp(-0.5 * x_values**2) / math.sqrt(2.0 * math.pi)


def gauss_legendre_on_interval(lower: float, upper: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Map Legendre nodes and weights from [-1, 1] to [lower, upper]."""
    nodes, weights = np.polynomial.legendre.leggauss(degree)
    scaled_nodes = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    scaled_weights = 0.5 * (upper - lower) * weights
    return scaled_nodes, scaled_weights


def build_integrated_beta_polynomial(
    cauchy_product: np.ndarray,
    deterministic_decay: np.ndarray,
    gaussian_bridge_std: np.ndarray,
    forward_variance: np.ndarray,
    normalization: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute the beta coefficients of VIX_T^2 as a polynomial in X_T."""
    polynomial_length = (len(cauchy_product) + 1) // 2
    beta_terms = []

    for power_index in range(0, 2 * polynomial_length - 1):
        k_values = np.arange(power_index, 2 * polynomial_length - 1)
        moment_factor = (
            gaussian_bridge_std ** (k_values[:, None] - power_index)
            * ((k_values[:, None] - power_index - 1) % 2)
            * np.array([double_factorial(value) for value in k_values - power_index - 1], dtype=float)[:, None]
            * comb(k_values, power_index)[:, None]
        )
        beta_slice = moment_factor * deterministic_decay[None, :] ** power_index * cauchy_product[k_values][:, None]
        beta_terms.append(np.sum(beta_slice, axis=0))

    beta_array = np.array(beta_terms, dtype=float) * forward_variance[None, :] / normalization[None, :]
    return np.sum((beta_array[:, :-1] + beta_array[:, 1:]) * 0.5, axis=1) * dt


def build_constant_h_transition_terms(
    model_parameters: ConstantHModelParameters,
    maturity: float,
    n_steps: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Closed-form transition quantities for the constant-H VIX formula."""
    mean_reversion = compute_constant_h_mean_reversion(model_parameters)
    diffusion_scale = compute_constant_h_vol_of_vol_scale(model_parameters)
    delta = VIX_WINDOW_YEARS
    u_grid = np.linspace(maturity, maturity + delta, n_steps + 1)
    dt = delta / n_steps

    deterministic_decay = np.exp(-mean_reversion * (u_grid - maturity))
    gaussian_bridge_std = diffusion_scale * np.sqrt(
        1.0 / (2.0 * mean_reversion) * (1.0 - np.exp(-2.0 * mean_reversion * (u_grid - maturity)))
    )
    factor_std_u = diffusion_scale * np.sqrt(
        1.0 / (2.0 * mean_reversion) * (1.0 - np.exp(-2.0 * mean_reversion * u_grid))
    )
    factor_std_t = float(
        diffusion_scale * math.sqrt(1.0 / (2.0 * mean_reversion) * (1.0 - math.exp(-2.0 * mean_reversion * maturity)))
    )
    return u_grid, dt, deterministic_decay, gaussian_bridge_std, factor_std_u, factor_std_t


def build_time_dependent_h_transition_terms(
    model_parameters: TimeDependentHModelParameters,
    maturity: float,
    n_steps: int,
    inner_quadrature_degree: int = 100,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Closed-form transition quantities for the time-dependent-H VIX formula."""
    delta = VIX_WINDOW_YEARS
    u_grid = np.linspace(maturity, maturity + delta, n_steps + 1)
    dt = delta / n_steps

    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(inner_quadrature_degree)
    exponent_t = compute_time_dependent_exponent(np.array([maturity]), model_parameters)[0]
    exponent_u = compute_time_dependent_exponent(u_grid, model_parameters)
    deterministic_decay = np.exp(-exponent_u + exponent_t)

    inner_times_u = 0.5 * (quadrature_nodes[None, :] + 1.0) * u_grid[:, None]
    inner_weights_u = 0.5 * quadrature_weights[None, :] * u_grid[:, None]
    inner_exponent_u = compute_time_dependent_exponent(inner_times_u, model_parameters)
    inner_h_u = evaluate_time_dependent_h_values(inner_times_u, model_parameters)
    inner_diffusion_u = model_parameters.epsilon ** (inner_h_u - 0.5)
    factor_std_u = np.sqrt(
        np.exp(-2.0 * exponent_u)
        * np.sum(np.exp(2.0 * inner_exponent_u) * inner_diffusion_u**2 * inner_weights_u, axis=1)
    )
    factor_std_t = float(factor_std_u[0])

    inner_times_diff = 0.5 * (quadrature_nodes[None, :] + 1.0) * (u_grid - maturity)[:, None] + maturity
    inner_weights_diff = 0.5 * quadrature_weights[None, :] * (u_grid - maturity)[:, None]
    inner_exponent_diff = compute_time_dependent_exponent(inner_times_diff, model_parameters)
    inner_h_diff = evaluate_time_dependent_h_values(inner_times_diff, model_parameters)
    inner_diffusion_diff = model_parameters.epsilon ** (inner_h_diff - 0.5)
    gaussian_bridge_std = np.sqrt(
        np.exp(-2.0 * exponent_u)
        * np.sum(np.exp(2.0 * inner_exponent_diff) * inner_diffusion_diff**2 * inner_weights_diff, axis=1)
    )
    return u_grid, dt, deterministic_decay, gaussian_bridge_std, factor_std_u, factor_std_t


def compute_vix_future_and_call_prices_from_beta(
    beta_polynomial: np.ndarray,
    factor_std_at_maturity: float,
    maturity: float,
    strike_percentages: np.ndarray,
    quadrature_degree: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Price the VIX future and a full smile once VIX_T^2 polynomial is known."""
    delta = VIX_WINDOW_YEARS
    gaussian_nodes, gaussian_weights = gauss_legendre_on_interval(-8.0, 8.0, quadrature_degree)
    density = gaussian_density(gaussian_nodes)
    vix_values = np.sqrt(
        np.maximum(
            evaluate_polynomial_with_horner(beta_polynomial[::-1], factor_std_at_maturity * gaussian_nodes) / delta,
            0.0,
        )
    )
    future = float(np.sum(density * vix_values * gaussian_weights))
    strikes = future * np.asarray(strike_percentages, dtype=float)
    option_prices = np.sum(
        density[None, :] * np.maximum(vix_values[None, :] - strikes[:, None], 0.0) * gaussian_weights[None, :],
        axis=1,
    )
    return future, strikes, option_prices


def price_vix_smile_analytic(
    experiment: ExperimentInput,
    maturity: float,
    strike_percentages: np.ndarray,
    quadrature_degree: int | None = None,
    n_steps: int | None = None,
    inner_quadrature_degree: int = 100,
) -> VIXSmileResult:
    """Analytic VIX smile pricing from the polynomial representation of VIX^2."""
    degree = experiment.monte_carlo.quadrature_degree if quadrature_degree is None else quadrature_degree
    steps = experiment.monte_carlo.vix_n_steps if n_steps is None else n_steps
    polynomial_vector = experiment.model_parameters.polynomial_vector()

    if isinstance(experiment.model_parameters, ConstantHModelParameters):
        transition_terms = build_constant_h_transition_terms(experiment.model_parameters, maturity, steps)
    else:
        transition_terms = build_time_dependent_h_transition_terms(
            experiment.model_parameters,
            maturity,
            steps,
            inner_quadrature_degree=inner_quadrature_degree,
        )

    u_grid, dt, deterministic_decay, gaussian_bridge_std, factor_std_u, factor_std_t = transition_terms
    cauchy_product = np.convolve(polynomial_vector, polynomial_vector)
    normalization = compute_polynomial_second_moment(polynomial_vector, factor_std_u)
    forward_variance = evaluate_forward_variance_curve(u_grid, experiment.forward_curve)
    beta_polynomial = build_integrated_beta_polynomial(
        cauchy_product,
        deterministic_decay,
        gaussian_bridge_std,
        forward_variance,
        normalization,
        dt,
    )
    future, strikes, option_prices = compute_vix_future_and_call_prices_from_beta(
        beta_polynomial,
        factor_std_t,
        maturity,
        np.asarray(strike_percentages, dtype=float),
        degree,
    )
    return VIXSmileResult(
        maturity=maturity,
        future=100.0 * future,
        strikes=100.0 * strikes,
        strike_perc=np.asarray(strike_percentages, dtype=float),
        option_prices=100.0 * option_prices,
        implied_volatility=implied_volatility_vector(option_prices, future, strikes, maturity),
    )
