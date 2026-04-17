"""Proxy xi0(t) construction utilities built from assembled option data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

try:
    from .csv_assembler import AssembledOptionData, assemble_option_chains, load_option_chains_from_directory
except ImportError:  # pragma: no cover
    from csv_assembler import AssembledOptionData, assemble_option_chains, load_option_chains_from_directory


@dataclass
class Xi0ComputationResult:
    quote_date: pd.Timestamp
    spot: float
    option_chain: pd.DataFrame
    file_metadata: pd.DataFrame
    atm_term_structure: pd.DataFrame
    xi0_step_curve: pd.DataFrame
    xi0_step: Callable[[object], object]
    xi0_smooth: Callable[[object], object]


def extract_atm_term_structure(option_chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    positive_maturity_chain = option_chain.loc[option_chain["ttm_years"] > 0.0].copy()
    if positive_maturity_chain.empty:
        raise ValueError("The option chain does not contain any strictly positive maturity.")

    atm_term_structure = (
        positive_maturity_chain.assign(
            distance_to_spot=(positive_maturity_chain["strike"] - spot).abs()
        )
        .sort_values(
            [
                "expiration",
                "distance_to_spot",
                "quote_completeness",
                "total_open_interest",
                "total_volume",
                "mean_bid_ask_spread",
                "strike",
            ],
            ascending=[True, True, False, False, False, True, True],
        )
        .groupby("expiration", as_index=False)
        .first()
    )

    atm_term_structure["atm_iv"] = 0.5 * (
        atm_term_structure["call_iv"] + atm_term_structure["put_iv"]
    )
    atm_term_structure["total_variance"] = (
        atm_term_structure["atm_iv"] ** 2 * atm_term_structure["ttm_years"]
    )

    return atm_term_structure[
        [
            "expiration",
            "ttm_days",
            "ttm_years",
            "strike",
            "call_symbol",
            "put_symbol",
            "source_file",
            "call_iv",
            "put_iv",
            "atm_iv",
            "total_variance",
            "distance_to_spot",
            "total_open_interest",
            "total_volume",
        ]
    ].sort_values("ttm_years").reset_index(drop=True)


def build_xi0_step_curve(
    atm_term_structure: pd.DataFrame,
    include_zero_interval: bool = False,
) -> tuple[pd.DataFrame, Callable[[object], object]]:
    if atm_term_structure.empty:
        raise ValueError("atm_term_structure is empty.")

    curve = atm_term_structure[["expiration", "ttm_years", "total_variance"]].copy()
    curve["t_left_years"] = curve["ttm_years"].shift(1)
    curve["w_left"] = curve["total_variance"].shift(1)

    if include_zero_interval:
        first_row = pd.DataFrame(
            {
                "expiration": [curve.iloc[0]["expiration"]],
                "ttm_years": [curve.iloc[0]["ttm_years"]],
                "total_variance": [curve.iloc[0]["total_variance"]],
                "t_left_years": [0.0],
                "w_left": [0.0],
            }
        )
        curve = pd.concat([first_row, curve.iloc[1:]], ignore_index=True)
    else:
        curve = curve.dropna().copy()

    curve["xi0"] = (
        (curve["total_variance"] - curve["w_left"])
        / (curve["ttm_years"] - curve["t_left_years"])
    ).clip(lower=0.0)

    xi0_step_curve = curve[
        ["expiration", "t_left_years", "ttm_years", "xi0"]
    ].rename(columns={"ttm_years": "t_right_years"}).reset_index(drop=True)
    xi0_step_curve["t_mid_years"] = 0.5 * (
        xi0_step_curve["t_left_years"] + xi0_step_curve["t_right_years"]
    )
    xi0_step_curve["t_left_days"] = xi0_step_curve["t_left_years"] * 365.0
    xi0_step_curve["t_right_days"] = xi0_step_curve["t_right_years"] * 365.0
    xi0_step_curve["t_mid_days"] = xi0_step_curve["t_mid_years"] * 365.0

    t_left = xi0_step_curve["t_left_years"].to_numpy()
    t_right = xi0_step_curve["t_right_years"].to_numpy()
    xi0_values = xi0_step_curve["xi0"].to_numpy()

    def xi0_step(t: object) -> object:
        time_grid = np.asarray(t, dtype=float)
        indices = np.searchsorted(t_right, time_grid, side="right")
        indices = np.clip(indices, 0, len(xi0_values) - 1)

        values = xi0_values[indices]
        values = np.where(time_grid < t_left[0], xi0_values[0], values)
        return values if values.ndim > 0 else float(values)

    return xi0_step_curve, xi0_step


def build_xi0_smooth_function(
    xi0_step_curve: pd.DataFrame,
) -> Callable[[object], object]:
    if xi0_step_curve.empty:
        raise ValueError("xi0_step_curve is empty.")

    knot_t = np.concatenate(
        (
            [xi0_step_curve["t_left_years"].iloc[0]],
            xi0_step_curve["t_mid_years"].to_numpy(),
            [xi0_step_curve["t_right_years"].iloc[-1]],
        )
    )
    knot_xi0 = np.concatenate(
        (
            [xi0_step_curve["xi0"].iloc[0]],
            xi0_step_curve["xi0"].to_numpy(),
            [xi0_step_curve["xi0"].iloc[-1]],
        )
    )

    interpolator = PchipInterpolator(knot_t, knot_xi0, extrapolate=True)
    t_min = float(knot_t[0])
    t_max = float(knot_t[-1])
    left_value = float(knot_xi0[0])
    right_value = float(knot_xi0[-1])

    def xi0_smooth(t: object) -> object:
        time_grid = np.asarray(t, dtype=float)
        values = interpolator(time_grid)
        values = np.clip(values, 0.0, None)
        values = np.where(time_grid < t_min, left_value, values)
        values = np.where(time_grid > t_max, right_value, values)
        return values if values.ndim > 0 else float(values)

    return xi0_smooth


def compute_xi0_from_assembled_data(
    assembled_data: AssembledOptionData,
    include_zero_interval: bool = False,
) -> Xi0ComputationResult:
    atm_term_structure = extract_atm_term_structure(
        option_chain=assembled_data.option_chain,
        spot=assembled_data.spot,
    )
    xi0_step_curve, xi0_step = build_xi0_step_curve(
        atm_term_structure=atm_term_structure,
        include_zero_interval=include_zero_interval,
    )
    xi0_smooth = build_xi0_smooth_function(xi0_step_curve=xi0_step_curve)

    return Xi0ComputationResult(
        quote_date=assembled_data.quote_date,
        spot=assembled_data.spot,
        option_chain=assembled_data.option_chain,
        file_metadata=assembled_data.file_metadata,
        atm_term_structure=atm_term_structure,
        xi0_step_curve=xi0_step_curve,
        xi0_step=xi0_step,
        xi0_smooth=xi0_smooth,
    )


def compute_xi0_from_csvs(
    csv_paths: Sequence[str | Path],
    require_same_quote_date: bool = True,
    include_zero_interval: bool = False,
) -> Xi0ComputationResult:
    assembled_data = assemble_option_chains(
        csv_paths=csv_paths,
        require_same_quote_date=require_same_quote_date,
    )
    return compute_xi0_from_assembled_data(
        assembled_data=assembled_data,
        include_zero_interval=include_zero_interval,
    )


def compute_xi0_from_directory(
    directory: str | Path,
    pattern: str = "*.csv",
    require_same_quote_date: bool = True,
    include_zero_interval: bool = False,
) -> Xi0ComputationResult:
    assembled_data = load_option_chains_from_directory(
        directory=directory,
        pattern=pattern,
        require_same_quote_date=require_same_quote_date,
    )
    return compute_xi0_from_assembled_data(
        assembled_data=assembled_data,
        include_zero_interval=include_zero_interval,
    )


def sample_xi0_curves(
    xi0_result: Xi0ComputationResult,
    sample_days: Sequence[float],
) -> pd.DataFrame:
    sample_days = np.asarray(sample_days, dtype=float)
    sample_years = sample_days / 365.0

    return pd.DataFrame(
        {
            "t_days": sample_days,
            "t_years": sample_years,
            "xi0_step": xi0_result.xi0_step(sample_years),
            "xi0_smooth": xi0_result.xi0_smooth(sample_years),
        }
    )


def plot_xi0_curves(
    xi0_result: Xi0ComputationResult,
    use_days: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    x_left = "t_left_days" if use_days else "t_left_years"
    x_right = "t_right_days" if use_days else "t_right_years"
    x_mid = "t_mid_days" if use_days else "t_mid_years"
    ttm_col = "ttm_days" if use_days else "ttm_years"
    xlabel = "maturity t (days)" if use_days else "maturity t (years)"

    t_dense = np.linspace(
        xi0_result.xi0_step_curve["t_left_years"].iloc[0],
        xi0_result.xi0_step_curve["t_right_years"].iloc[-1],
        500,
    )
    x_dense = t_dense * 365.0 if use_days else t_dense

    ax.step(
        xi0_result.xi0_step_curve[x_right],
        xi0_result.xi0_step_curve["xi0"],
        where="post",
        linewidth=2.2,
        color="#1f77b4",
        label="xi0 step",
    )
    ax.plot(
        x_dense,
        xi0_result.xi0_smooth(t_dense),
        linewidth=2.8,
        color="#d62728",
        label="xi0 smooth",
    )
    ax.scatter(
        xi0_result.xi0_step_curve[x_mid],
        xi0_result.xi0_step_curve["xi0"],
        s=28,
        color="black",
        alpha=0.75,
        zorder=3,
        label="smoothing knots",
    )

    for maturity in xi0_result.atm_term_structure[ttm_col]:
        ax.axvline(maturity, color="0.9", linewidth=0.8, zorder=0)

    ax.set_title("Proxy forward variance curve xi0(t)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("xi0(t)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax

