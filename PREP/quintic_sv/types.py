from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstantHParams:
    rho: float
    H: float
    eps: float
    alpha0: float
    alpha1: float
    alpha3: float
    alpha5: float

    def coefficients(self) -> np.ndarray:
        return np.array(
            [self.alpha0, self.alpha1, 0.0, self.alpha3, 0.0, self.alpha5],
            dtype=float,
        )


@dataclass(frozen=True)
class TimeDependentHParams:
    rho: float
    h0: float
    h_inf: float
    h_kappa: float
    eps: float
    alpha0: float
    alpha1: float
    alpha3: float
    alpha5: float

    def coefficients(self) -> np.ndarray:
        return np.array(
            [self.alpha0, self.alpha1, 0.0, self.alpha3, 0.0, self.alpha5],
            dtype=float,
        )


@dataclass(frozen=True)
class SplineScenario:
    name: str
    spot: float
    params: ConstantHParams
    forward_node_times: np.ndarray
    forward_node_values: np.ndarray
    spx_maturities: np.ndarray
    vix_maturities: np.ndarray
    spx_n_steps: int
    spx_n_sims: int
    vix_n_steps: int
    quadrature_degree: int
    seed: int
    market_seed: int
    vix_log_moneyness_min: float = -0.1
    vix_log_moneyness_max: float = 1.0
    vix_num_strikes: int = 50


@dataclass(frozen=True)
class ParametricScenario:
    name: str
    spot: float
    params: ConstantHParams
    curve_a: float
    curve_b: float
    curve_c: float
    spx_maturities: np.ndarray
    vix_maturities: np.ndarray
    spx_n_steps: int
    spx_n_sims: int
    vix_n_steps: int
    quadrature_degree: int
    seed: int
    market_seed: int
    vix_log_moneyness_min: float = -0.1
    vix_log_moneyness_max: float = 1.0
    vix_num_strikes: int = 50


@dataclass(frozen=True)
class TimeDependentScenario:
    name: str
    spot: float
    params: TimeDependentHParams
    forward_node_times: np.ndarray
    forward_node_values: np.ndarray
    spx_maturities: np.ndarray
    vix_maturities: np.ndarray
    spx_n_steps: int
    spx_n_sims: int
    vix_n_steps: int
    quadrature_degree: int
    seed: int
    market_seed: int
    vix_log_moneyness_min: float = -0.1
    vix_log_moneyness_max: float = 1.0
    vix_num_strikes: int = 50


@dataclass
class SPXSmileResult:
    maturity: float
    strikes: np.ndarray
    log_moneyness: np.ndarray
    prices: np.ndarray
    model_iv: np.ndarray
    lower_iv: np.ndarray
    upper_iv: np.ndarray
    market_bid_iv: np.ndarray | None = None
    market_ask_iv: np.ndarray | None = None


@dataclass
class VIXSmileResult:
    maturity: float
    future: float
    strikes: np.ndarray
    strike_perc: np.ndarray
    prices: np.ndarray
    model_iv: np.ndarray
    market_future: float | None = None
    market_bid_iv: np.ndarray | None = None
    market_ask_iv: np.ndarray | None = None


@dataclass
class ForwardCurveComparison:
    node_times: np.ndarray
    calibrated_nodes: np.ndarray
    stripped_nodes: np.ndarray
    grid_times: np.ndarray
    calibrated_curve: np.ndarray
    stripped_curve: np.ndarray
