from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quintic_sv.configs import (
    paper_constant_h_spline_scenario,
    paper_parametric_medium_scenario,
    paper_parametric_short_scenario,
    paper_time_dependent_scenario,
)
from quintic_sv.paper_workflow import build_paper_figure_bundle
from quintic_sv.plots import plot_forward_curve, plot_joint_smiles, plot_time_dependent_h
from quintic_sv.spx_deep_learning import load_spx_surrogate_model
from quintic_sv.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate honest paper-style figures from the published inputs.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "figures"), help="Directory where PNG files are written.")
    parser.add_argument("--spx-engine", choices=("mc", "dl"), default="mc", help="Use Monte Carlo or the deep learning surrogate for SPX smiles.")
    parser.add_argument("--surrogate-model", default=None, help="Path to a .npz model saved by train_spx_surrogate.py when --spx-engine=dl.")
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Scale numerical budgets. Use 1.0 for the closest paper-style reproduction, or a smaller value for a faster run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    surrogate_model = None
    if args.spx_engine == "dl":
        if args.surrogate_model is None:
            raise ValueError("--surrogate-model is required when --spx-engine=dl.")
        surrogate_model = load_spx_surrogate_model(args.surrogate_model)

    bundle_12 = build_paper_figure_bundle(
        paper_constant_h_spline_scenario(),
        spx_engine=args.spx_engine,
        surrogate_model=surrogate_model,
        budget_scale=args.budget_scale,
    )
    bundle_3 = build_paper_figure_bundle(
        paper_parametric_short_scenario(),
        spx_engine=args.spx_engine,
        surrogate_model=surrogate_model,
        budget_scale=args.budget_scale,
    )
    bundle_4 = build_paper_figure_bundle(
        paper_parametric_medium_scenario(),
        spx_engine=args.spx_engine,
        surrogate_model=surrogate_model,
        budget_scale=args.budget_scale,
    )
    bundle_567 = build_paper_figure_bundle(
        paper_time_dependent_scenario(),
        spx_engine=args.spx_engine,
        surrogate_model=surrogate_model,
        budget_scale=args.budget_scale,
    )

    fig1 = plot_joint_smiles(bundle_12.spx_smiles, bundle_12.vix_smiles, "side_by_side", output_dir / "figure_1_constant_h_joint_smiles.png")
    fig2 = plot_forward_curve(bundle_12.forward_curve_trace, "Forward variance curve", output_dir / "figure_2_constant_h_forward_curve.png")
    fig3 = plot_joint_smiles(bundle_3.spx_smiles, bundle_3.vix_smiles, "side_by_side", output_dir / "figure_3_parametric_short_joint_smiles.png")
    fig4 = plot_joint_smiles(bundle_4.spx_smiles, bundle_4.vix_smiles, "side_by_side", output_dir / "figure_4_parametric_medium_joint_smiles.png")
    fig5 = plot_joint_smiles(bundle_567.spx_smiles, bundle_567.vix_smiles, "stacked", output_dir / "figure_5_time_dependent_joint_smiles.png")
    fig6 = plot_forward_curve(bundle_567.forward_curve_trace, "Forward variance curve", output_dir / "figure_6_time_dependent_forward_curve.png")
    if bundle_567.h_trace is None:
        raise RuntimeError("The time-dependent scenario should always produce an H(t) trace.")
    fig7 = plot_time_dependent_h(bundle_567.h_trace, output_dir / "figure_7_time_dependent_h.png")

    for figure in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        figure.clf()

    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
