from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def black_scholes_call_price(
    spot: np.ndarray | float,
    sigma: np.ndarray | float,
    maturity: float,
    strike: np.ndarray | float,
) -> np.ndarray:
    """Black-Scholes call price with zero rates and zero dividends."""
    spot_arr = np.asarray(spot, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    strike_arr = np.asarray(strike, dtype=float)

    intrinsic = np.maximum(spot_arr - strike_arr, 0.0)
    if maturity <= 0:
        return intrinsic

    safe_sigma = np.maximum(sigma_arr, 1e-12)
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(np.maximum(spot_arr, 1e-14) / strike_arr) + 0.5 * safe_sigma**2 * maturity) / (
        safe_sigma * sqrt_t
    )
    d2 = d1 - safe_sigma * sqrt_t
    price = spot_arr * ndtr(d1) - strike_arr * ndtr(d2)
    return np.where(sigma_arr <= 1e-12, intrinsic, price)


def implied_volatility_call(price: float, spot: float, strike: float, maturity: float) -> float:
    """Invert the Black-Scholes call price by 1D root finding."""
    intrinsic = max(spot - strike, 0.0)
    upper_price = max(spot - 1e-12, intrinsic + 1e-12)
    clipped_price = min(max(price, intrinsic + 1e-12), upper_price)

    if maturity <= 0 or clipped_price <= intrinsic + 1e-10:
        return 0.0

    def objective(vol: float) -> float:
        return float(black_scholes_call_price(spot, vol, maturity, strike) - clipped_price)

    try:
        return float(brentq(objective, 1e-8, 5.0, maxiter=200))
    except ValueError:
        try:
            return float(brentq(objective, 1e-8, 10.0, maxiter=200))
        except ValueError:
            return float("nan")


def implied_volatility_vector(
    prices: Iterable[float],
    spot: float,
    strikes: Iterable[float],
    maturity: float,
) -> np.ndarray:
    """Vectorized wrapper around implied_volatility_call."""
    strikes_arr = np.asarray(list(strikes), dtype=float)
    prices_arr = np.asarray(list(prices), dtype=float)
    return np.array(
        [
            implied_volatility_call(price, spot, strike, maturity)
            for price, strike in zip(prices_arr, strikes_arr, strict=True)
        ],
        dtype=float,
    )
