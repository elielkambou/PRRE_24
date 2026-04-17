from __future__ import annotations

import math

import numpy as np

from .black import black_scholes_call_price, implied_volatility_vector
from .curves import evaluate_forward_variance_curve
from .factor_process import simulate_factor_paths_for_model
from .polynomial_volatility import build_volatility_paths_from_factor_paths
from .types import ExperimentInput, MonteCarloSettings, SPXSmileResult
from .utils import generate_antithetic_normals


def build_volatility_paths_for_experiment(
    experiment: ExperimentInput,
    maturity: float,
    normals: np.ndarray,
    inner_quadrature_degree: int = 100,
) -> np.ndarray:
    """Simulate sigma_t along all Monte Carlo paths for one maturity."""
    factor_simulation = simulate_factor_paths_for_model(
        experiment.model_parameters,
        maturity,
        normals,
        inner_quadrature_degree=inner_quadrature_degree,
    )
    forward_variance = evaluate_forward_variance_curve(factor_simulation.times, experiment.forward_curve)
    return build_volatility_paths_from_factor_paths(
        experiment.model_parameters.polynomial_vector(),
        factor_simulation.factor_paths,
        factor_simulation.factor_std,
        forward_variance,
    )


def simulate_terminal_log_spot(
    spot: float,
    rho: float,
    maturity: float,
    volatility_paths: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Euler step the correlated log spot once sigma_t has been constructed."""
    dt = maturity / (normals.shape[0] - 1)
    log_spot = np.full(volatility_paths.shape[1], math.log(spot), dtype=float)

    for step in range(normals.shape[0] - 1):
        correlated_sigma = rho * volatility_paths[step]
        log_spot = log_spot - 0.5 * dt * correlated_sigma**2 + math.sqrt(dt) * correlated_sigma * normals[step + 1]

    return log_spot


def compute_control_variate_call_prices(
    spot: float,
    rho: float,
    maturity: float,
    strikes: np.ndarray,
    terminal_log_spot: np.ndarray,
    integrated_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Price calls with the same control variate idea as the paper notebook."""
    terminal_spot = np.exp(terminal_log_spot)
    total_paths = terminal_spot.size
    q_value = float(np.max(integrated_variance) + 1e-9)

    x_payoff = black_scholes_call_price(
        terminal_spot,
        np.sqrt(np.maximum((1.0 - rho**2) * integrated_variance / maturity, 0.0)),
        maturity,
        strikes.reshape(-1, 1),
    ).T
    y_payoff = black_scholes_call_price(
        terminal_spot,
        np.sqrt(np.maximum(rho**2 * (q_value - integrated_variance) / maturity, 0.0)),
        maturity,
        strikes.reshape(-1, 1),
    ).T
    expected_y = black_scholes_call_price(
        spot,
        math.sqrt(max(rho**2 * q_value / maturity, 0.0)),
        maturity,
        strikes.reshape(-1, 1),
    ).T

    control_coefficients = []
    for strike_index in range(strikes.shape[0]):
        covariance = np.cov(x_payoff[:, strike_index] + 10.0, y_payoff[:, strike_index] + 10.0)[0, 1]
        variance = np.cov(x_payoff[:, strike_index] + 10.0, y_payoff[:, strike_index] + 10.0)[1, 1]
        if covariance < 1e-8 or variance < 1e-8:
            coefficient = 1e-40
        else:
            coefficient = float(np.nan_to_num(covariance / variance, nan=1e-40))
        control_coefficients.append(min(coefficient, 2.0))

    control_array = np.array(control_coefficients, dtype=float)
    adjusted_payoffs = x_payoff - control_array * (y_payoff - expected_y)
    mean_prices = np.mean(adjusted_payoffs, axis=0)
    std_prices = np.std(adjusted_payoffs, axis=0)
    confidence_half_width = 1.96 * std_prices / np.sqrt(total_paths)
    return mean_prices, confidence_half_width


def price_spx_smile_with_monte_carlo(
    experiment: ExperimentInput,
    maturity: float,
    strikes: np.ndarray,
    monte_carlo_settings: MonteCarloSettings | None = None,
    random_seed: int | None = None,
) -> SPXSmileResult:
    """Reference SPX smile pricing engine based on Monte Carlo."""
    settings = monte_carlo_settings or experiment.monte_carlo
    seed = settings.random_seed if random_seed is None else random_seed
    normals = generate_antithetic_normals(settings.spx_n_steps, settings.spx_n_base_paths, seed)

    volatility_paths = build_volatility_paths_for_experiment(experiment, maturity, normals)
    terminal_log_spot = simulate_terminal_log_spot(
        experiment.spot,
        experiment.model_parameters.rho,
        maturity,
        volatility_paths,
        normals,
    )
    integrated_variance = np.sum(volatility_paths**2 * (maturity / settings.spx_n_steps), axis=0)
    option_prices, confidence_half_width = compute_control_variate_call_prices(
        experiment.spot,
        experiment.model_parameters.rho,
        maturity,
        np.asarray(strikes, dtype=float),
        terminal_log_spot,
        integrated_variance,
    )
    return SPXSmileResult(
        maturity=maturity,
        strikes=np.asarray(strikes, dtype=float),
        log_moneyness=np.log(np.asarray(strikes, dtype=float) / experiment.spot),
        option_prices=option_prices,
        implied_volatility=implied_volatility_vector(option_prices, experiment.spot, strikes, maturity),
        engine_name="monte_carlo",
        confidence_interval_half_width=confidence_half_width,
    )
