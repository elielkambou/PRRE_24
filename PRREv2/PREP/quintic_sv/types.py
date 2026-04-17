from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


@dataclass(frozen=True)
class QuinticPolynomialCoefficients:
    """Coefficients of p(x) = a0 + a1 x + a3 x^3 + a5 x^5."""

    alpha0: float
    alpha1: float
    alpha3: float
    alpha5: float

    def as_polynomial_vector(self) -> np.ndarray:
        """Return the full polynomial vector including zero even terms."""
        return np.array([self.alpha0, self.alpha1, 0.0, self.alpha3, 0.0, self.alpha5], dtype=float)


@dataclass(frozen=True)
class ConstantHModelParameters:
    """Model inputs when H is constant through time."""

    rho: float
    h_value: float
    epsilon: float
    polynomial: QuinticPolynomialCoefficients

    def polynomial_vector(self) -> np.ndarray:
        return self.polynomial.as_polynomial_vector()


@dataclass(frozen=True)
class TimeDependentHModelParameters:
    """Model inputs when H(t) = h0 e^{-k t} + h_inf (1 - e^{-k t})."""

    rho: float
    h0: float
    h_inf: float
    h_kappa: float
    epsilon: float
    polynomial: QuinticPolynomialCoefficients

    def polynomial_vector(self) -> np.ndarray:
        return self.polynomial.as_polynomial_vector()


ModelParameters: TypeAlias = ConstantHModelParameters | TimeDependentHModelParameters


@dataclass(frozen=True)
class SplineForwardVarianceCurve:
    """Forward variance curve represented by spline nodes."""

    node_times: np.ndarray
    node_values: np.ndarray


@dataclass(frozen=True)
class ParametricForwardVarianceCurve:
    """Forward variance curve xi_0(t) = a exp(-b t) + c (1 - exp(-b t))."""

    a: float
    b: float
    c: float


ForwardCurveParameters: TypeAlias = SplineForwardVarianceCurve | ParametricForwardVarianceCurve


@dataclass(frozen=True)
class SmileGridSettings:
    """Maturities and strike conventions used to build smiles."""

    spx_maturities: np.ndarray
    vix_maturities: np.ndarray
    vix_log_moneyness_min: float = -0.1
    vix_log_moneyness_max: float = 1.0
    vix_num_strikes: int = 50


@dataclass(frozen=True)
class MonteCarloSettings:
    """Numerical settings for the reference pricing engines."""

    spx_n_steps: int
    spx_n_base_paths: int
    vix_n_steps: int
    quadrature_degree: int
    random_seed: int


@dataclass(frozen=True)
class ExperimentInput:
    """Full input needed to reproduce one paper experiment."""

    name: str
    spot: float
    model_parameters: ModelParameters
    forward_curve: ForwardCurveParameters
    smile_grid: SmileGridSettings
    monte_carlo: MonteCarloSettings


@dataclass
class FactorPathSimulation:
    """Time grid, simulated factor paths X_t, and theoretical factor std."""

    times: np.ndarray
    factor_paths: np.ndarray
    factor_std: np.ndarray


@dataclass
class SPXSmileResult:
    """SPX smile returned by either the Monte Carlo or the deep surrogate."""

    maturity: float
    strikes: np.ndarray
    log_moneyness: np.ndarray
    option_prices: np.ndarray
    implied_volatility: np.ndarray
    engine_name: str
    confidence_interval_half_width: np.ndarray | None = None


@dataclass
class VIXSmileResult:
    """VIX smile priced analytically from the polynomial VIX representation."""

    maturity: float
    future: float
    strikes: np.ndarray
    strike_perc: np.ndarray
    option_prices: np.ndarray
    implied_volatility: np.ndarray


@dataclass
class CurveTrace:
    """Simple curve container used by the plotting layer."""

    label: str
    times: np.ndarray
    values: np.ndarray


@dataclass
class PaperFigureBundle:
    """Group of objects plotted together for one published scenario."""

    experiment: ExperimentInput
    spx_smiles: list[SPXSmileResult]
    vix_smiles: list[VIXSmileResult]
    forward_curve_trace: CurveTrace
    h_trace: CurveTrace | None = None
