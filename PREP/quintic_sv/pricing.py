from __future__ import annotations

import math

import numpy as np
from scipy.special import comb

from .black import black_scholes_call_price, implied_volatility_vector
from .curves import evaluate_parametric_forward_curve, evaluate_spline_forward_curve
from .model import normalization_variance, simulate_xt_grid_constant_h, simulate_xt_grid_time_dependent_h, time_dependent_exponent
from .types import SPXSmileResult, VIXSmileResult
from .utils import VIX_WINDOW_YEARS, double_factorial, horner_vector


def _build_volatility_paths(
    polynomial_coeffs: np.ndarray,
    xt_paths: np.ndarray,
    std_x: np.ndarray,
    forward_curve: np.ndarray,
) -> np.ndarray:
    raw_polynomial = horner_vector(polynomial_coeffs[::-1], xt_paths)
    normal_var = normalization_variance(polynomial_coeffs, std_x)
    return np.sqrt(forward_curve)[:, None] * raw_polynomial / np.sqrt(normal_var)[:, None]


def _control_variate_prices(
    rho: float,
    maturity: float,
    spot: float,
    strikes: np.ndarray,
    volatility: np.ndarray,
    normals: np.ndarray,
    n_sims: int,
) -> tuple[np.ndarray, np.ndarray]:
    dt = maturity / (normals.shape[0] - 1)
    log_s = np.full(volatility.shape[1], math.log(spot), dtype=float)
    for step in range(normals.shape[0] - 1):
        sigma_rho = rho * volatility[step]
        log_s = log_s - 0.5 * dt * sigma_rho**2 + math.sqrt(dt) * sigma_rho * normals[step + 1]

    terminal_spot = np.exp(log_s)
    integrated_variance = np.sum(volatility[:-1] ** 2 * dt, axis=0)
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

    control_coeffs = []
    for index in range(strikes.shape[0]):
        covariance = np.cov(x_payoff[:, index] + 10.0, y_payoff[:, index] + 10.0)[0, 1]
        variance = np.cov(x_payoff[:, index] + 10.0, y_payoff[:, index] + 10.0)[1, 1]
        if covariance < 1e-8 or variance < 1e-8:
            coeff = 1e-40
        else:
            coeff = float(np.nan_to_num(covariance / variance, nan=1e-40))
        control_coeffs.append(min(coeff, 2.0))

    control_coeffs_arr = np.array(control_coeffs, dtype=float)
    call_prices = x_payoff - control_coeffs_arr * (y_payoff - expected_y)
    mean_prices = np.average(call_prices, axis=0)
    std_prices = np.std(call_prices, axis=0)
    conf_scale = 1.96 * std_prices / np.sqrt(2 * n_sims)
    return mean_prices, conf_scale


def price_spx_smile_constant_h_spline(
    rho: float,
    H: float,
    eps: float,
    maturity: float,
    polynomial_coeffs: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    forward_node_times: np.ndarray,
    forward_node_values: np.ndarray,
    normals: np.ndarray,
    n_sims: int,
) -> SPXSmileResult:
    times, xt_paths, std_x = simulate_xt_grid_constant_h(H, eps, maturity, normals)
    forward_curve = evaluate_spline_forward_curve(times, forward_node_times, forward_node_values)
    volatility = _build_volatility_paths(polynomial_coeffs, xt_paths, std_x, forward_curve)
    prices, conf_scale = _control_variate_prices(rho, maturity, spot, strikes, volatility, normals, n_sims)
    return SPXSmileResult(
        maturity=maturity,
        strikes=strikes,
        log_moneyness=np.log(strikes / spot),
        prices=prices,
        model_iv=implied_volatility_vector(prices, spot, strikes, maturity),
        lower_iv=implied_volatility_vector(np.maximum(prices - conf_scale, 1e-12), spot, strikes, maturity),
        upper_iv=implied_volatility_vector(prices + conf_scale, spot, strikes, maturity),
    )


def price_spx_smile_constant_h_parametric(
    rho: float,
    H: float,
    eps: float,
    maturity: float,
    polynomial_coeffs: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    curve_a: float,
    curve_b: float,
    curve_c: float,
    normals: np.ndarray,
    n_sims: int,
) -> SPXSmileResult:
    times, xt_paths, std_x = simulate_xt_grid_constant_h(H, eps, maturity, normals)
    forward_curve = evaluate_parametric_forward_curve(times, curve_a, curve_b, curve_c)
    volatility = _build_volatility_paths(polynomial_coeffs, xt_paths, std_x, forward_curve)
    prices, conf_scale = _control_variate_prices(rho, maturity, spot, strikes, volatility, normals, n_sims)
    return SPXSmileResult(
        maturity=maturity,
        strikes=strikes,
        log_moneyness=np.log(strikes / spot),
        prices=prices,
        model_iv=implied_volatility_vector(prices, spot, strikes, maturity),
        lower_iv=implied_volatility_vector(np.maximum(prices - conf_scale, 1e-12), spot, strikes, maturity),
        upper_iv=implied_volatility_vector(prices + conf_scale, spot, strikes, maturity),
    )


def price_spx_smile_time_dependent_h_spline(
    rho: float,
    h0: float,
    h_inf: float,
    h_kappa: float,
    eps: float,
    maturity: float,
    polynomial_coeffs: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    forward_node_times: np.ndarray,
    forward_node_values: np.ndarray,
    normals: np.ndarray,
    n_sims: int,
) -> SPXSmileResult:
    times, xt_paths, std_x = simulate_xt_grid_time_dependent_h(h0, h_inf, h_kappa, eps, maturity, normals)
    forward_curve = evaluate_spline_forward_curve(times, forward_node_times, forward_node_values)
    volatility = _build_volatility_paths(polynomial_coeffs, xt_paths, std_x, forward_curve)
    prices, conf_scale = _control_variate_prices(rho, maturity, spot, strikes, volatility, normals, n_sims)
    return SPXSmileResult(
        maturity=maturity,
        strikes=strikes,
        log_moneyness=np.log(strikes / spot),
        prices=prices,
        model_iv=implied_volatility_vector(prices, spot, strikes, maturity),
        lower_iv=implied_volatility_vector(np.maximum(prices - conf_scale, 1e-12), spot, strikes, maturity),
        upper_iv=implied_volatility_vector(prices + conf_scale, spot, strikes, maturity),
    )


def _gaussian_density(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _gauss_legendre(lower: float, upper: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(degree)
    return 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower), 0.5 * (upper - lower) * weights


def _integrated_beta_polynomial(
    cauchy_product: np.ndarray,
    exp_det: np.ndarray,
    std_g: np.ndarray,
    forward_curve: np.ndarray,
    normal_var: np.ndarray,
    dt: float,
) -> np.ndarray:
    poly_len = (len(cauchy_product) + 1) // 2
    beta_terms = []
    for power_i in range(0, 2 * poly_len - 1):
        k_array = np.arange(power_i, 2 * poly_len - 1)
        moment_factor = (
            std_g ** (k_array[:, None] - power_i)
            * ((k_array[:, None] - power_i - 1) % 2)
            * np.array([double_factorial(value) for value in k_array - power_i - 1], dtype=float)[:, None]
            * comb(k_array, power_i)[:, None]
        )
        beta_temp = moment_factor * exp_det[None, :] ** power_i * cauchy_product[k_array][:, None]
        beta_terms.append(np.sum(beta_temp, axis=0))

    beta = np.array(beta_terms, dtype=float) * forward_curve[None, :] / normal_var[None, :]
    return np.sum((beta[:, :-1] + beta[:, 1:]) * 0.5, axis=1) * dt


def price_vix_smile_constant_h_spline(
    H: float,
    eps: float,
    maturity: float,
    polynomial_coeffs: np.ndarray,
    strike_perc: np.ndarray,
    forward_node_times: np.ndarray,
    forward_node_values: np.ndarray,
    quadrature_degree: int = 400,
    n_steps: int = 200,
) -> VIXSmileResult:
    kappa_tilde = (0.5 - H) / eps
    eta_tilde = eps ** (H - 0.5)
    delta = VIX_WINDOW_YEARS
    u_grid = np.linspace(maturity, maturity + delta, n_steps + 1)
    dt = delta / n_steps

    y_nodes, y_weights = _gauss_legendre(-8.0, 8.0, quadrature_degree)
    exp_det = np.exp(-kappa_tilde * (u_grid - maturity))
    std_g = eta_tilde * np.sqrt(1.0 / (2.0 * kappa_tilde) * (1.0 - np.exp(-2.0 * kappa_tilde * (u_grid - maturity))))
    std_x_u = eta_tilde * np.sqrt(1.0 / (2.0 * kappa_tilde) * (1.0 - np.exp(-2.0 * kappa_tilde * u_grid)))
    std_x_t = eta_tilde * math.sqrt(1.0 / (2.0 * kappa_tilde) * (1.0 - math.exp(-2.0 * kappa_tilde * maturity)))

    cauchy_product = np.convolve(polynomial_coeffs, polynomial_coeffs)
    normal_var = normalization_variance(polynomial_coeffs, std_x_u)
    forward_curve = evaluate_spline_forward_curve(u_grid, forward_node_times, forward_node_values)
    beta_integrated = _integrated_beta_polynomial(cauchy_product, exp_det, std_g, forward_curve, normal_var, dt)

    vix_values = np.sqrt(np.maximum(horner_vector(beta_integrated[::-1], std_x_t * y_nodes) / delta, 0.0))
    density = _gaussian_density(y_nodes)
    future = np.sum(density * vix_values * y_weights)
    strikes = future * strike_perc
    prices = np.sum(density[None, :] * np.maximum(vix_values[None, :] - strikes[:, None], 0.0) * y_weights[None, :], axis=1)
    return VIXSmileResult(
        maturity=maturity,
        future=100.0 * future,
        strikes=100.0 * strikes,
        strike_perc=strike_perc,
        prices=100.0 * prices,
        model_iv=implied_volatility_vector(prices, future, strikes, maturity),
    )


def price_vix_smile_constant_h_parametric(
    H: float,
    eps: float,
    maturity: float,
    polynomial_coeffs: np.ndarray,
    strike_perc: np.ndarray,
    curve_a: float,
    curve_b: float,
    curve_c: float,
    quadrature_degree: int = 400,
    n_steps: int = 200,
) -> VIXSmileResult:
    kappa_tilde = (0.5 - H) / eps
    eta_tilde = eps ** (H - 0.5)
    delta = VIX_WINDOW_YEARS
    u_grid = np.linspace(maturity, maturity + delta, n_steps + 1)
    dt = delta / n_steps

    y_nodes, y_weights = _gauss_legendre(-8.0, 8.0, quadrature_degree)
    exp_det = np.exp(-kappa_tilde * (u_grid - maturity))
    std_g = eta_tilde * np.sqrt(1.0 / (2.0 * kappa_tilde) * (1.0 - np.exp(-2.0 * kappa_tilde * (u_grid - maturity))))
    std_x_u = eta_tilde * np.sqrt(1.0 / (2.0 * kappa_tilde) * (1.0 - np.exp(-2.0 * kappa_tilde * u_grid)))
    std_x_t = eta_tilde * math.sqrt(1.0 / (2.0 * kappa_tilde) * (1.0 - math.exp(-2.0 * kappa_tilde * maturity)))

    cauchy_product = np.convolve(polynomial_coeffs, polynomial_coeffs)
    normal_var = normalization_variance(polynomial_coeffs, std_x_u)
    forward_curve = evaluate_parametric_forward_curve(u_grid, curve_a, curve_b, curve_c)
    beta_integrated = _integrated_beta_polynomial(cauchy_product, exp_det, std_g, forward_curve, normal_var, dt)

    vix_values = np.sqrt(np.maximum(horner_vector(beta_integrated[::-1], std_x_t * y_nodes) / delta, 0.0))
    density = _gaussian_density(y_nodes)
    future = np.sum(density * vix_values * y_weights)
    strikes = future * strike_perc
    prices = np.sum(density[None, :] * np.maximum(vix_values[None, :] - strikes[:, None], 0.0) * y_weights[None, :], axis=1)
    return VIXSmileResult(
        maturity=maturity,
        future=100.0 * future,
        strikes=100.0 * strikes,
        strike_perc=strike_perc,
        prices=100.0 * prices,
        model_iv=implied_volatility_vector(prices, future, strikes, maturity),
    )


def price_vix_smile_time_dependent_h_spline(
    h0: float,
    h_inf: float,
    h_kappa: float,
    eps: float,
    maturity: float,
    polynomial_coeffs: np.ndarray,
    strike_perc: np.ndarray,
    forward_node_times: np.ndarray,
    forward_node_values: np.ndarray,
    quadrature_degree: int = 400,
    n_steps: int = 200,
    inner_quadrature_degree: int = 100,
) -> VIXSmileResult:
    delta = VIX_WINDOW_YEARS
    u_grid = np.linspace(maturity, maturity + delta, n_steps + 1)
    dt = delta / n_steps
    y_nodes, y_weights = _gauss_legendre(-8.0, 8.0, quadrature_degree)
    x_inner, w_inner = np.polynomial.legendre.leggauss(inner_quadrature_degree)

    exponent_x_t = time_dependent_exponent(np.array([maturity]), h0, h_inf, h_kappa, eps)[0]
    exponent_u = time_dependent_exponent(u_grid, h0, h_inf, h_kappa, eps)
    exp_det = np.exp(-exponent_u + exponent_x_t)

    inner_times_u = 0.5 * (x_inner[None, :] + 1.0) * u_grid[:, None]
    inner_weights_u = 0.5 * w_inner[None, :] * u_grid[:, None]
    inner_exp_u = time_dependent_exponent(inner_times_u, h0, h_inf, h_kappa, eps)
    inner_h_u = h0 * np.exp(-h_kappa * inner_times_u) + h_inf * (1.0 - np.exp(-h_kappa * inner_times_u))
    inner_eta_u = eps ** (inner_h_u - 0.5)
    std_x_u = np.sqrt(np.exp(-2.0 * exponent_u) * np.sum(np.exp(2.0 * inner_exp_u) * inner_eta_u**2 * inner_weights_u, axis=1))
    std_x_t = float(std_x_u[0])

    inner_times_diff = 0.5 * (x_inner[None, :] + 1.0) * (u_grid - maturity)[:, None] + maturity
    inner_weights_diff = 0.5 * w_inner[None, :] * (u_grid - maturity)[:, None]
    inner_exp_diff = time_dependent_exponent(inner_times_diff, h0, h_inf, h_kappa, eps)
    inner_h_diff = h0 * np.exp(-h_kappa * inner_times_diff) + h_inf * (1.0 - np.exp(-h_kappa * inner_times_diff))
    inner_eta_diff = eps ** (inner_h_diff - 0.5)
    std_g = np.sqrt(
        np.exp(-2.0 * exponent_u)
        * np.sum(np.exp(2.0 * inner_exp_diff) * inner_eta_diff**2 * inner_weights_diff, axis=1)
    )

    cauchy_product = np.convolve(polynomial_coeffs, polynomial_coeffs)
    normal_var = normalization_variance(polynomial_coeffs, std_x_u)
    forward_curve = evaluate_spline_forward_curve(u_grid, forward_node_times, forward_node_values)
    beta_integrated = _integrated_beta_polynomial(cauchy_product, exp_det, std_g, forward_curve, normal_var, dt)

    vix_values = np.sqrt(np.maximum(horner_vector(beta_integrated[::-1], std_x_t * y_nodes) / delta, 0.0))
    density = _gaussian_density(y_nodes)
    future = np.sum(density * vix_values * y_weights)
    strikes = future * strike_perc
    prices = np.sum(density[None, :] * np.maximum(vix_values[None, :] - strikes[:, None], 0.0) * y_weights[None, :], axis=1)
    return VIXSmileResult(
        maturity=maturity,
        future=100.0 * future,
        strikes=100.0 * strikes,
        strike_perc=strike_perc,
        prices=100.0 * prices,
        model_iv=implied_volatility_vector(prices, future, strikes, maturity),
    )
