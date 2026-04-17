from __future__ import annotations

import numpy as np

from .configs import (
    paper_constant_h_spline_scenario,
    paper_parametric_medium_scenario,
    paper_parametric_short_scenario,
    paper_time_dependent_scenario,
)
from .curves import evaluate_forward_variance_curve, evaluate_time_dependent_h_values
from .spx_deep_learning import SPXSurrogateModel, price_spx_smile_with_surrogate
from .spx_monte_carlo import price_spx_smile_with_monte_carlo
from .types import CurveTrace, ExperimentInput, MonteCarloSettings, PaperFigureBundle
from .utils import VIX_WINDOW_YEARS, default_spx_log_moneyness_range, scaled_int
from .vix_analytic import price_vix_smile_analytic


def build_spx_strike_grid(spot: float, maturity: float, strike_count: int = 30) -> np.ndarray:
    """Paper-style SPX strike grid from maturity-dependent log-moneyness bounds."""
    lower_bound, upper_bound = default_spx_log_moneyness_range(maturity)
    return spot * np.exp(np.linspace(lower_bound, upper_bound, strike_count))


def build_vix_strike_percent_grid(experiment: ExperimentInput) -> np.ndarray:
    """Paper-style VIX strike percentages."""
    return np.exp(
        np.linspace(
            experiment.smile_grid.vix_log_moneyness_min,
            experiment.smile_grid.vix_log_moneyness_max,
            experiment.smile_grid.vix_num_strikes,
        )
    )


def scale_monte_carlo_settings(settings: MonteCarloSettings, budget_scale: float) -> MonteCarloSettings:
    """Reduce Monte Carlo budgets for notebook demos or fast smoke tests."""
    return MonteCarloSettings(
        spx_n_steps=scaled_int(settings.spx_n_steps, budget_scale, 64),
        spx_n_base_paths=scaled_int(settings.spx_n_base_paths, budget_scale, 600),
        vix_n_steps=scaled_int(settings.vix_n_steps, budget_scale, 80),
        quadrature_degree=scaled_int(settings.quadrature_degree, budget_scale, 120),
        random_seed=settings.random_seed,
    )


def price_spx_smile_with_requested_engine(
    experiment: ExperimentInput,
    maturity: float,
    strikes: np.ndarray,
    spx_engine: str,
    surrogate_model: SPXSurrogateModel | None,
    monte_carlo_settings: MonteCarloSettings,
) -> object:
    """Switch between the exact Monte Carlo engine and the neural surrogate."""
    if spx_engine == "mc":
        return price_spx_smile_with_monte_carlo(experiment, maturity, strikes, monte_carlo_settings=monte_carlo_settings)
    if spx_engine == "dl":
        if surrogate_model is None:
            raise ValueError("A trained surrogate model is required when spx_engine='dl'.")
        return price_spx_smile_with_surrogate(experiment, maturity, strikes, surrogate_model)
    raise ValueError(f"Unknown SPX engine: {spx_engine}")


def build_forward_curve_trace(experiment: ExperimentInput, point_count: int = 400) -> CurveTrace:
    """Sample the input forward variance curve on a dense grid."""
    max_time = max(
        float(np.max(experiment.smile_grid.spx_maturities)),
        float(np.max(experiment.smile_grid.vix_maturities)) + VIX_WINDOW_YEARS,
    )
    times = np.linspace(0.0, max_time, point_count)
    values = evaluate_forward_variance_curve(times, experiment.forward_curve)
    return CurveTrace(label="forward_variance_curve", times=times, values=values)


def build_h_trace(experiment: ExperimentInput, point_count: int = 400) -> CurveTrace | None:
    """Return H(t) when the scenario uses a time-dependent H."""
    if not hasattr(experiment.model_parameters, "h0"):
        return None
    max_time = max(
        float(np.max(experiment.smile_grid.spx_maturities)),
        float(np.max(experiment.smile_grid.vix_maturities)),
    )
    times = np.linspace(0.0, max_time, point_count)
    values = evaluate_time_dependent_h_values(times, experiment.model_parameters)
    return CurveTrace(label="time_dependent_h", times=times, values=values)


def build_paper_figure_bundle(
    experiment: ExperimentInput,
    spx_engine: str = "mc",
    surrogate_model: SPXSurrogateModel | None = None,
    budget_scale: float = 1.0,
) -> PaperFigureBundle:
    """Price all smiles required by one paper scenario and group the outputs."""
    scaled_settings = scale_monte_carlo_settings(experiment.monte_carlo, budget_scale)
    vix_strike_percentages = build_vix_strike_percent_grid(experiment)

    spx_smiles = []
    for maturity in experiment.smile_grid.spx_maturities:
        strikes = build_spx_strike_grid(experiment.spot, float(maturity))
        spx_smiles.append(
            price_spx_smile_with_requested_engine(
                experiment,
                float(maturity),
                strikes,
                spx_engine=spx_engine,
                surrogate_model=surrogate_model,
                monte_carlo_settings=scaled_settings,
            )
        )

    vix_smiles = []
    for maturity in experiment.smile_grid.vix_maturities:
        vix_smiles.append(
            price_vix_smile_analytic(
                experiment,
                float(maturity),
                vix_strike_percentages,
                quadrature_degree=scaled_settings.quadrature_degree,
                n_steps=scaled_settings.vix_n_steps,
            )
        )

    return PaperFigureBundle(
        experiment=experiment,
        spx_smiles=spx_smiles,
        vix_smiles=vix_smiles,
        forward_curve_trace=build_forward_curve_trace(experiment),
        h_trace=build_h_trace(experiment),
    )


def build_all_paper_experiments() -> list[ExperimentInput]:
    """Return the four published paper scenarios in a deterministic order."""
    return [
        paper_constant_h_spline_scenario(),
        paper_parametric_short_scenario(),
        paper_parametric_medium_scenario(),
        paper_time_dependent_scenario(),
    ]
