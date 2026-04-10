from __future__ import annotations

import numpy as np

from .types import ConstantHParams, ParametricScenario, SplineScenario, TimeDependentHParams, TimeDependentScenario


def paper_constant_h_spline_scenario() -> SplineScenario:
    return SplineScenario(
        name="figure_1_and_2",
        spot=100.0,
        params=ConstantHParams(-0.6842593, -0.03583785, 1 / 52, 0.59069477, 0.99999991, 0.28927803, 0.05491437),
        forward_node_times=np.array(
            [0.0, 0.02601012, 0.03696174, 0.05065128, 0.06297186, 0.07529244, 0.08898198, 0.09993361, 0.10540942, 0.11636105, 0.13552640, 0.16427442, 0.21218779, 0.25736326],
            dtype=float,
        ),
        forward_node_values=np.array(
            [0.00950134, 0.00799541, 0.00514354, 0.01079057, 0.02396814, 0.00793446, 0.01301627, 0.02776016, 0.00222057, 0.01965447, 0.01085834, 0.02437202, 0.01278166, 0.01881186],
            dtype=float,
        ),
        spx_maturities=np.array([0.03011698, 0.08213721, 0.18343977, 0.24093582], dtype=float),
        vix_maturities=np.array([0.02464116, 0.06297186, 0.08213721, 0.15879861], dtype=float),
        spx_n_steps=800,
        spx_n_sims=20_000,
        vix_n_steps=200,
        quadrature_degree=400,
        seed=42,
        market_seed=1701,
    )


def paper_parametric_short_scenario() -> ParametricScenario:
    return ParametricScenario(
        name="figure_3",
        spot=100.0,
        params=ConstantHParams(-0.73157011, -0.13815974, 1 / 52, 0.81685253, 0.27397169, 0.17173771, 0.00360953),
        curve_a=0.0084409,
        curve_b=2.04363437,
        curve_c=0.04406909,
        spx_maturities=np.array([0.02464116, 0.08213721], dtype=float),
        vix_maturities=np.array([0.02464116], dtype=float),
        spx_n_steps=400,
        spx_n_sims=10_000,
        vix_n_steps=200,
        quadrature_degree=400,
        seed=42,
        market_seed=2203,
    )


def paper_parametric_medium_scenario() -> ParametricScenario:
    return ParametricScenario(
        name="figure_4",
        spot=100.0,
        params=ConstantHParams(-0.7001, 0.141, 1 / 52, 0.7558, 1.0, 0.0885, 0.4421),
        curve_a=0.012,
        curve_b=2.027,
        curve_c=0.033,
        spx_maturities=np.array([0.14510907, 0.24093582], dtype=float),
        vix_maturities=np.array([0.15879861], dtype=float),
        spx_n_steps=400,
        spx_n_sims=10_000,
        vix_n_steps=200,
        quadrature_degree=400,
        seed=42,
        market_seed=3307,
        vix_log_moneyness_max=1.3,
    )


def paper_time_dependent_scenario() -> TimeDependentScenario:
    return TimeDependentScenario(
        name="figure_5_to_7",
        spot=100.0,
        params=TimeDependentHParams(-7.46560529e-01, 3.17620635e-01, -1.36650102e00, 1.19935418e01, 1.35941796e-01, 1.0e-10, 2.66441061e-02, 2.51336785e-01, 5.87934269e-05),
        forward_node_times=np.array(
            [0.0, 0.03285488, 0.05612709, 0.08761302, 0.12594372, 0.19302244, 0.25736326, 0.29569396, 0.33402465, 0.37235535, 0.41205500, 0.53662977, 0.77756559, 1.03629780, 1.16],
            dtype=float,
        ),
        forward_node_values=np.array(
            [0.01031423, 0.01107315, 0.01234894, 0.01404261, 0.01595505, 0.01783993, 0.02130452, 0.02413495, 0.02455308, 0.02683603, 0.02818511, 0.03133947, 0.03852319, 0.03261153, 0.03186],
            dtype=float,
        ),
        spx_maturities=np.array([0.03011698, 0.08213721, 0.18343977, 0.24093582, 0.68173884, 1.16087257], dtype=float),
        vix_maturities=np.array([0.02464116, 0.08213721, 0.15879861, 0.23546, 0.3121214, 0.40794814], dtype=float),
        spx_n_steps=400,
        spx_n_sims=10_000,
        vix_n_steps=200,
        quadrature_degree=400,
        seed=42,
        market_seed=4411,
    )
