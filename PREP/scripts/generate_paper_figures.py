from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quintic_sv.paper_figures import build_figure_1_bundle, build_figure_3_bundle, build_figure_4_bundle, build_figure_5_bundle
from quintic_sv.plots import plot_forward_curve_comparison, plot_joint_smiles, plot_time_dependent_h
from quintic_sv.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the main Quintic-SV paper-style figures.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "figures"), help="Directory where PNG files are written.")
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Scale Monte Carlo budgets. Use 1.0 for the closest paper-style reproduction, or a smaller value for a faster run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    bundle_12 = build_figure_1_bundle(budget_scale=args.budget_scale)
    fig1 = plot_joint_smiles(bundle_12["spx_smiles"], bundle_12["vix_smiles"], "side_by_side", output_dir / "figure_1_joint_smiles.png")
    fig2 = plot_forward_curve_comparison(bundle_12["forward_curve"], "Forward variance curve", output_dir / "figure_2_forward_curve.png")
    fig1.clf()
    fig2.clf()

    bundle_3 = build_figure_3_bundle(budget_scale=args.budget_scale)
    fig3 = plot_joint_smiles(bundle_3["spx_smiles"], bundle_3["vix_smiles"], "side_by_side", output_dir / "figure_3_parametric_short.png")
    fig3.clf()

    bundle_4 = build_figure_4_bundle(budget_scale=args.budget_scale)
    fig4 = plot_joint_smiles(bundle_4["spx_smiles"], bundle_4["vix_smiles"], "side_by_side", output_dir / "figure_4_parametric_medium.png")
    fig4.clf()

    bundle_567 = build_figure_5_bundle(budget_scale=args.budget_scale)
    fig5 = plot_joint_smiles(bundle_567["spx_smiles"], bundle_567["vix_smiles"], "stacked", output_dir / "figure_5_time_dependent_joint.png")
    fig6 = plot_forward_curve_comparison(bundle_567["forward_curve"], "Forward variance curve", output_dir / "figure_6_forward_curve_long.png")
    fig7 = plot_time_dependent_h(bundle_567["h_times"], bundle_567["h_values"], output_dir / "figure_7_time_dependent_h.png")
    fig5.clf()
    fig6.clf()
    fig7.clf()

    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
