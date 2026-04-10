from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .types import ForwardCurveComparison, SPXSmileResult, VIXSmileResult
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
        if smile.market_bid_iv is not None:
            axis.plot(smile.log_moneyness, smile.market_bid_iv, ".", color="royalblue", markersize=3)
        if smile.market_ask_iv is not None:
            axis.plot(smile.log_moneyness, smile.market_ask_iv, ".", color="tomato", markersize=3)
        axis.plot(smile.log_moneyness, smile.model_iv, color="green", linewidth=1.0)
        axis.set_title(f"T = {maturity_to_days(smile.maturity)} days", fontsize=9)
        axis.set_xlabel(r"log moneyness $\log(K/S_0)$", fontsize=8)
        axis.tick_params(labelsize=7)

    for axis in axes_spx[len(spx_smiles) :]:
        axis.axis("off")

    for axis, smile in zip(axes_vix, vix_smiles, strict=True):
        if smile.market_bid_iv is not None:
            axis.plot(smile.strikes, smile.market_bid_iv, ".", color="royalblue", markersize=3)
        if smile.market_ask_iv is not None:
            axis.plot(smile.strikes, smile.market_ask_iv, ".", color="tomato", markersize=3)
        axis.plot(smile.strikes, smile.model_iv, color="green", linewidth=1.0)
        if smile.market_future is not None:
            axis.axvline(smile.market_future, color="black", linewidth=0.8)
        axis.set_title(f"T = {maturity_to_days(smile.maturity)} days", fontsize=9)
        axis.set_xlabel("strike", fontsize=8)
        axis.tick_params(labelsize=7)

    for axis in axes_vix[len(vix_smiles) :]:
        axis.axis("off")

    _save(fig, outpath)
    return fig


def plot_forward_curve_comparison(
    curves: ForwardCurveComparison,
    title: str = "Forward variance curve",
    outpath: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.6), constrained_layout=True)
    ax.set_title(title)
    ax.plot(curves.grid_times, curves.stripped_curve, color="royalblue", linewidth=1.0, label="forward curve mkt")
    ax.plot(curves.grid_times, curves.calibrated_curve, color="green", linestyle="--", linewidth=1.0, label="forward curve after calib.")
    ax.plot(curves.node_times, curves.stripped_nodes, "o", color="royalblue", markersize=3)
    ax.plot(curves.node_times, curves.calibrated_nodes, "o", color="green", markersize=3)
    ax.set_xlabel("time")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, outpath)
    return fig


def plot_time_dependent_h(
    times: np.ndarray,
    values: np.ndarray,
    outpath: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.4), constrained_layout=True)
    ax.set_title("Time dependent H")
    ax.plot(times, values, color="tab:blue", linewidth=1.2)
    ax.set_xlabel("time")
    _save(fig, outpath)
    return fig
