from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .types import CurveTrace, SPXSmileResult, VIXSmileResult
from .utils import maturity_to_days


def _grid_shape(count: int) -> tuple[int, int]:
    if count <= 1:
        return (1, 1)
    if count == 2:
        return (1, 2)
    if count == 3:
        return (1, 3)
    if count == 4:
        return (2, 2)
    if count <= 6:
        return (2, 3)
    return (2, 4)


def _as_flat_axes(fig: plt.Figure, grid_spec, rows: int, cols: int) -> list[plt.Axes]:
    axes = []
    for index in range(rows * cols):
        row = index // cols
        col = index % cols
        axes.append(fig.add_subplot(grid_spec[row, col]))
    return axes


def _save(fig: plt.Figure, outpath: str | Path | None) -> None:
    if outpath is not None:
        fig.savefig(outpath, dpi=180, bbox_inches="tight")


def plot_joint_smiles(
    spx_smiles: list[SPXSmileResult],
    vix_smiles: list[VIXSmileResult],
    orientation: str = "side_by_side",
    outpath: str | Path | None = None,
) -> plt.Figure:
    """Plot SPX and VIX smiles without inventing market bid/ask points."""
    rows_spx, cols_spx = _grid_shape(len(spx_smiles))
    rows_vix, cols_vix = _grid_shape(len(vix_smiles))

    figsize = (14, 6) if orientation == "side_by_side" else (12, 8)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    if orientation == "side_by_side":
        outer = fig.add_gridspec(1, 2, width_ratios=(1.1, 1.1))
        grid_spx = outer[0].subgridspec(rows_spx, cols_spx)
        grid_vix = outer[1].subgridspec(rows_vix, cols_vix)
        fig.text(0.24, 0.995, "SPX implied volatility", ha="center", va="top", fontsize=10)
        fig.text(0.76, 0.995, "VIX implied volatility", ha="center", va="top", fontsize=10)
    else:
        outer = fig.add_gridspec(2, 1, height_ratios=(1.15, 1.0))
        grid_spx = outer[0].subgridspec(rows_spx, cols_spx)
        grid_vix = outer[1].subgridspec(rows_vix, cols_vix)
        fig.text(0.5, 0.995, "SPX implied volatility", ha="center", va="top", fontsize=10)
        fig.text(0.5, 0.505, "VIX implied volatility", ha="center", va="top", fontsize=10)

    axes_spx = _as_flat_axes(fig, grid_spx, rows_spx, cols_spx)
    axes_vix = _as_flat_axes(fig, grid_vix, rows_vix, cols_vix)

    for axis, smile in zip(axes_spx, spx_smiles, strict=True):
        axis.plot(smile.log_moneyness, smile.implied_volatility, color="green", linewidth=1.2)
        axis.set_title(f"T = {maturity_to_days(smile.maturity)} days ({smile.engine_name})", fontsize=9)
        axis.set_xlabel(r"log moneyness $\log(K/S_0)$", fontsize=8)
        axis.set_ylabel("implied vol", fontsize=8)
        axis.tick_params(labelsize=7)

    for axis in axes_spx[len(spx_smiles) :]:
        axis.axis("off")

    for axis, smile in zip(axes_vix, vix_smiles, strict=True):
        axis.plot(smile.strikes, smile.implied_volatility, color="green", linewidth=1.2)
        axis.axvline(smile.future, color="black", linewidth=0.8)
        axis.set_title(f"T = {maturity_to_days(smile.maturity)} days", fontsize=9)
        axis.set_xlabel("strike", fontsize=8)
        axis.set_ylabel("implied vol", fontsize=8)
        axis.tick_params(labelsize=7)

    for axis in axes_vix[len(vix_smiles) :]:
        axis.axis("off")

    _save(fig, outpath)
    return fig


def plot_forward_curve(
    trace: CurveTrace,
    title: str = "Forward variance curve",
    outpath: str | Path | None = None,
) -> plt.Figure:
    """Plot the forward variance curve used as model input."""
    fig, ax = plt.subplots(figsize=(10, 2.6), constrained_layout=True)
    ax.set_title(title)
    ax.plot(trace.times, trace.values, color="green", linewidth=1.2)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\xi_0(t)$")
    _save(fig, outpath)
    return fig


def plot_time_dependent_h(
    trace: CurveTrace,
    outpath: str | Path | None = None,
) -> plt.Figure:
    """Plot H(t) when the scenario uses the time-dependent extension."""
    fig, ax = plt.subplots(figsize=(10, 2.4), constrained_layout=True)
    ax.set_title("Time dependent H")
    ax.plot(trace.times, trace.values, color="tab:blue", linewidth=1.2)
    ax.set_xlabel("time")
    ax.set_ylabel("H(t)")
    _save(fig, outpath)
    return fig


def plot_spx_monte_carlo_vs_surrogate(
    monte_carlo_smile: SPXSmileResult,
    surrogate_smile: SPXSmileResult,
    outpath: str | Path | None = None,
) -> plt.Figure:
    """Directly compare the SPX Monte Carlo smile and the neural surrogate smile."""
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax.plot(monte_carlo_smile.log_moneyness, monte_carlo_smile.implied_volatility, color="black", linewidth=1.2, label="Monte Carlo")
    ax.plot(surrogate_smile.log_moneyness, surrogate_smile.implied_volatility, color="tab:orange", linewidth=1.2, label="Deep surrogate")
    ax.set_xlabel(r"log moneyness $\log(K/S_0)$")
    ax.set_ylabel("implied vol")
    ax.legend(loc="best")
    _save(fig, outpath)
    return fig
