from __future__ import annotations

import numpy as np

from .configs import (
    paper_constant_h_spline_scenario,
    paper_parametric_medium_scenario,
    paper_parametric_short_scenario,
    paper_time_dependent_scenario,
)
from .curves import evaluate_time_dependent_h
from .market import add_synthetic_market_to_spx, add_synthetic_market_to_vix, simulate_forward_curve_comparison
from .pricing import (
    price_spx_smile_constant_h_parametric,
    price_spx_smile_constant_h_spline,
    price_spx_smile_time_dependent_h_spline,
    price_vix_smile_constant_h_parametric,
    price_vix_smile_constant_h_spline,
    price_vix_smile_time_dependent_h_spline,
)
from .utils import generate_antithetic_normals, scaled_int, spx_log_moneyness_range


def _vix_strike_perc(vix_log_moneyness_min: float, vix_log_moneyness_max: float, count: int) -> np.ndarray:
    return np.exp(np.linspace(vix_log_moneyness_min, vix_log_moneyness_max, count))


def build_figure_1_bundle(budget_scale: float = 1.0) -> dict:
    scenario = paper_constant_h_spline_scenario()
    n_steps = scaled_int(scenario.spx_n_steps, budget_scale, 100)
    n_sims = scaled_int(scenario.spx_n_sims, budget_scale, 1_500)
    normals = generate_antithetic_normals(n_steps, n_sims, scenario.seed)
    coeffs = scenario.params.coefficients()

    spx_smiles = []
    for index, maturity in enumerate(scenario.spx_maturities):
        lm_min, lm_max = spx_log_moneyness_range(float(maturity))
        strikes = np.exp(np.linspace(lm_min, lm_max, 30)) * scenario.spot
        smile = price_spx_smile_constant_h_spline(
            scenario.params.rho,
            scenario.params.H,
            scenario.params.eps,
            float(maturity),
            coeffs,
            scenario.spot,
            strikes,
            scenario.forward_node_times,
            scenario.forward_node_values,
            normals,
            n_sims,
        )
        spx_smiles.append(add_synthetic_market_to_spx(smile, scenario.market_seed + index))

    strike_perc = _vix_strike_perc(scenario.vix_log_moneyness_min, scenario.vix_log_moneyness_max, scenario.vix_num_strikes)
    vix_smiles = []
    for index, maturity in enumerate(scenario.vix_maturities):
        smile = price_vix_smile_constant_h_spline(
            scenario.params.H,
            scenario.params.eps,
            float(maturity),
            coeffs,
            strike_perc,
            scenario.forward_node_times,
            scenario.forward_node_values,
            scenario.quadrature_degree,
            scenario.vix_n_steps,
        )
        vix_smiles.append(add_synthetic_market_to_vix(smile, scenario.market_seed + 100 + index))

    return {
        "scenario": scenario,
        "spx_smiles": spx_smiles,
        "vix_smiles": vix_smiles,
        "forward_curve": simulate_forward_curve_comparison(
            scenario.forward_node_times,
            scenario.forward_node_values,
            scenario.market_seed + 200,
        ),
    }


def build_figure_3_bundle(budget_scale: float = 1.0) -> dict:
    scenario = paper_parametric_short_scenario()
    n_steps = scaled_int(scenario.spx_n_steps, budget_scale, 100)
    n_sims = scaled_int(scenario.spx_n_sims, budget_scale, 1_500)
    normals = generate_antithetic_normals(n_steps, n_sims, scenario.seed)
    coeffs = scenario.params.coefficients()

    spx_smiles = []
    for index, maturity in enumerate(scenario.spx_maturities):
        lm_min, lm_max = spx_log_moneyness_range(float(maturity))
        strikes = np.exp(np.linspace(lm_min, lm_max, 30)) * scenario.spot
        smile = price_spx_smile_constant_h_parametric(
            scenario.params.rho,
            scenario.params.H,
            scenario.params.eps,
            float(maturity),
            coeffs,
            scenario.spot,
            strikes,
            scenario.curve_a,
            scenario.curve_b,
            scenario.curve_c,
            normals,
            n_sims,
        )
        spx_smiles.append(add_synthetic_market_to_spx(smile, scenario.market_seed + index))

    strike_perc = _vix_strike_perc(scenario.vix_log_moneyness_min, scenario.vix_log_moneyness_max, scenario.vix_num_strikes)
    vix_smiles = []
    for index, maturity in enumerate(scenario.vix_maturities):
        smile = price_vix_smile_constant_h_parametric(
            scenario.params.H,
            scenario.params.eps,
            float(maturity),
            coeffs,
            strike_perc,
            scenario.curve_a,
            scenario.curve_b,
            scenario.curve_c,
            scenario.quadrature_degree,
            scenario.vix_n_steps,
        )
        vix_smiles.append(add_synthetic_market_to_vix(smile, scenario.market_seed + 100 + index))

    return {"scenario": scenario, "spx_smiles": spx_smiles, "vix_smiles": vix_smiles}


def build_figure_4_bundle(budget_scale: float = 1.0) -> dict:
    scenario = paper_parametric_medium_scenario()
    n_steps = scaled_int(scenario.spx_n_steps, budget_scale, 100)
    n_sims = scaled_int(scenario.spx_n_sims, budget_scale, 1_500)
    normals = generate_antithetic_normals(n_steps, n_sims, scenario.seed)
    coeffs = scenario.params.coefficients()

    spx_smiles = []
    for index, maturity in enumerate(scenario.spx_maturities):
        lm_min, lm_max = spx_log_moneyness_range(float(maturity))
        strikes = np.exp(np.linspace(lm_min, lm_max, 30)) * scenario.spot
        smile = price_spx_smile_constant_h_parametric(
            scenario.params.rho,
            scenario.params.H,
            scenario.params.eps,
            float(maturity),
            coeffs,
            scenario.spot,
            strikes,
            scenario.curve_a,
            scenario.curve_b,
            scenario.curve_c,
            normals,
            n_sims,
        )
        spx_smiles.append(add_synthetic_market_to_spx(smile, scenario.market_seed + index))

    strike_perc = _vix_strike_perc(scenario.vix_log_moneyness_min, scenario.vix_log_moneyness_max, scenario.vix_num_strikes)
    vix_smiles = []
    for index, maturity in enumerate(scenario.vix_maturities):
        smile = price_vix_smile_constant_h_parametric(
            scenario.params.H,
            scenario.params.eps,
            float(maturity),
            coeffs,
            strike_perc,
            scenario.curve_a,
            scenario.curve_b,
            scenario.curve_c,
            scenario.quadrature_degree,
            scenario.vix_n_steps,
        )
        vix_smiles.append(add_synthetic_market_to_vix(smile, scenario.market_seed + 100 + index))

    return {"scenario": scenario, "spx_smiles": spx_smiles, "vix_smiles": vix_smiles}


def build_figure_5_bundle(budget_scale: float = 1.0) -> dict:
    scenario = paper_time_dependent_scenario()
    n_steps = scaled_int(scenario.spx_n_steps, budget_scale, 100)
    n_sims = scaled_int(scenario.spx_n_sims, budget_scale, 1_500)
    normals = generate_antithetic_normals(n_steps, n_sims, scenario.seed)
    coeffs = scenario.params.coefficients()

    spx_smiles = []
    for index, maturity in enumerate(scenario.spx_maturities):
        lm_min, lm_max = spx_log_moneyness_range(float(maturity))
        strikes = np.exp(np.linspace(lm_min, lm_max, 30)) * scenario.spot
        smile = price_spx_smile_time_dependent_h_spline(
            scenario.params.rho,
            scenario.params.h0,
            scenario.params.h_inf,
            scenario.params.h_kappa,
            scenario.params.eps,
            float(maturity),
            coeffs,
            scenario.spot,
            strikes,
            scenario.forward_node_times,
            scenario.forward_node_values,
            normals,
            n_sims,
        )
        spx_smiles.append(add_synthetic_market_to_spx(smile, scenario.market_seed + index))

    strike_perc = _vix_strike_perc(scenario.vix_log_moneyness_min, scenario.vix_log_moneyness_max, scenario.vix_num_strikes)
    vix_smiles = []
    for index, maturity in enumerate(scenario.vix_maturities):
        smile = price_vix_smile_time_dependent_h_spline(
            scenario.params.h0,
            scenario.params.h_inf,
            scenario.params.h_kappa,
            scenario.params.eps,
            float(maturity),
            coeffs,
            strike_perc,
            scenario.forward_node_times,
            scenario.forward_node_values,
            scenario.quadrature_degree,
            scenario.vix_n_steps,
        )
        vix_smiles.append(add_synthetic_market_to_vix(smile, scenario.market_seed + 100 + index))

    h_times = np.linspace(0.0, float(scenario.forward_node_times[-1]), 2_000)
    return {
        "scenario": scenario,
        "spx_smiles": spx_smiles,
        "vix_smiles": vix_smiles,
        "forward_curve": simulate_forward_curve_comparison(
            scenario.forward_node_times,
            scenario.forward_node_values,
            scenario.market_seed + 200,
        ),
        "h_times": h_times,
        "h_values": evaluate_time_dependent_h(
            h_times,
            scenario.params.h0,
            scenario.params.h_inf,
            scenario.params.h_kappa,
        ),
    }
