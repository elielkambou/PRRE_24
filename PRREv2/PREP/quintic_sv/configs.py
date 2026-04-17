from __future__ import annotations

import numpy as np

from .types import (
    ConstantHModelParameters,
    ExperimentInput,
    MonteCarloSettings,
    ParametricForwardVarianceCurve,
    QuinticPolynomialCoefficients,
    SmileGridSettings,
    SplineForwardVarianceCurve,
    TimeDependentHModelParameters,
)


def _default_smile_grid(spx_maturities: list[float], vix_maturities: list[float]) -> SmileGridSettings:
    return SmileGridSettings(
        spx_maturities=np.array(spx_maturities, dtype=float),
        vix_maturities=np.array(vix_maturities, dtype=float),
    )


def paper_constant_h_spline_scenario() -> ExperimentInput:
    """Published inputs used for Figures 1 and 2 of the paper."""
    return ExperimentInput(
        name="figure_1_and_2",
        spot=100.0,
        model_parameters=ConstantHModelParameters(
            rho=-0.6842593,
            h_value=-0.03583785,
            epsilon=1 / 52,
            polynomial=QuinticPolynomialCoefficients(0.59069477, 0.99999991, 0.28927803, 0.05491437),
        ),
        forward_curve=SplineForwardVarianceCurve(
            node_times=np.array(
                [0.0, 0.02601012, 0.03696174, 0.05065128, 0.06297186, 0.07529244, 0.08898198, 0.09993361, 0.10540942, 0.11636105, 0.13552640, 0.16427442, 0.21218779, 0.25736326],
                dtype=float,
            ),
            node_values=np.array(
                [0.00950134, 0.00799541, 0.00514354, 0.01079057, 0.02396814, 0.00793446, 0.01301627, 0.02776016, 0.00222057, 0.01965447, 0.01085834, 0.02437202, 0.01278166, 0.01881186],
                dtype=float,
            ),
        ),
        smile_grid=_default_smile_grid(
            spx_maturities=[0.03011698, 0.08213721, 0.18343977, 0.24093582],
            vix_maturities=[0.02464116, 0.06297186, 0.08213721, 0.15879861],
        ),
        monte_carlo=MonteCarloSettings(
            spx_n_steps=800,
            spx_n_base_paths=20_000,
            vix_n_steps=200,
            quadrature_degree=400,
            random_seed=42,
        ),
    )


def paper_parametric_short_scenario() -> ExperimentInput:
    """Published inputs used for Figure 3 of the paper."""
    return ExperimentInput(
        name="figure_3",
        spot=100.0,
        model_parameters=ConstantHModelParameters(
            rho=-0.73157011,
            h_value=-0.13815974,
            epsilon=1 / 52,
            polynomial=QuinticPolynomialCoefficients(0.81685253, 0.27397169, 0.17173771, 0.00360953),
        ),
        forward_curve=ParametricForwardVarianceCurve(a=0.0084409, b=2.04363437, c=0.04406909),
        smile_grid=_default_smile_grid(
            spx_maturities=[0.02464116, 0.08213721],
            vix_maturities=[0.02464116],
        ),
        monte_carlo=MonteCarloSettings(
            spx_n_steps=400,
            spx_n_base_paths=10_000,
            vix_n_steps=200,
            quadrature_degree=400,
            random_seed=42,
        ),
    )


def paper_parametric_medium_scenario() -> ExperimentInput:
    """Published inputs used for Figure 4 of the paper."""
    return ExperimentInput(
        name="figure_4",
        spot=100.0,
        model_parameters=ConstantHModelParameters(
            rho=-0.7001,
            h_value=0.141,
            epsilon=1 / 52,
            polynomial=QuinticPolynomialCoefficients(0.7558, 1.0, 0.0885, 0.4421),
        ),
        forward_curve=ParametricForwardVarianceCurve(a=0.012, b=2.027, c=0.033),
        smile_grid=SmileGridSettings(
            spx_maturities=np.array([0.14510907, 0.24093582], dtype=float),
            vix_maturities=np.array([0.15879861], dtype=float),
            vix_log_moneyness_min=-0.1,
            vix_log_moneyness_max=1.3,
            vix_num_strikes=50,
        ),
        monte_carlo=MonteCarloSettings(
            spx_n_steps=400,
            spx_n_base_paths=10_000,
            vix_n_steps=200,
            quadrature_degree=400,
            random_seed=42,
        ),
    )


def paper_time_dependent_scenario() -> ExperimentInput:
    """Published inputs used for Figures 5, 6 and 7 of the paper."""
    return ExperimentInput(
        name="figure_5_to_7",
        spot=100.0,
        model_parameters=TimeDependentHModelParameters(
            rho=-7.46560529e-01,
            h0=3.17620635e-01,
            h_inf=-1.36650102e00,
            h_kappa=1.19935418e01,
            epsilon=1.35941796e-01,
            polynomial=QuinticPolynomialCoefficients(1.0e-10, 2.66441061e-02, 2.51336785e-01, 5.87934269e-05),
        ),
        forward_curve=SplineForwardVarianceCurve(
            node_times=np.array(
                [0.0, 0.03285488, 0.05612709, 0.08761302, 0.12594372, 0.19302244, 0.25736326, 0.29569396, 0.33402465, 0.37235535, 0.41205500, 0.53662977, 0.77756559, 1.03629780, 1.16],
                dtype=float,
            ),
            node_values=np.array(
                [0.01031423, 0.01107315, 0.01234894, 0.01404261, 0.01595505, 0.01783993, 0.02130452, 0.02413495, 0.02455308, 0.02683603, 0.02818511, 0.03133947, 0.03852319, 0.03261153, 0.03186],
                dtype=float,
            ),
        ),
        smile_grid=_default_smile_grid(
            spx_maturities=[0.03011698, 0.08213721, 0.18343977, 0.24093582, 0.68173884, 1.16087257],
            vix_maturities=[0.02464116, 0.08213721, 0.15879861, 0.23546, 0.3121214, 0.40794814],
        ),
        monte_carlo=MonteCarloSettings(
            spx_n_steps=400,
            spx_n_base_paths=10_000,
            vix_n_steps=200,
            quadrature_degree=400,
            random_seed=42,
        ),
    )
