from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quintic_sv.utils import ensure_directory


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Reproduction du modele Quintic SV\n\n"
            "Ce notebook suit le papier `The quintic Ornstein-Uhlenbeck volatility model that jointly calibrates SPX & VIX smiles`.\n\n"
            "Objectif : reconstruire les courbes du papier avec les parametres publies, puis simuler les donnees de marche manquantes la ou le papier utilisait des donnees CBOE."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Structure du projet\n\n"
            "Le projet est separe en petits modules :\n\n"
            "- `quintic_sv/model.py` pour la dynamique de `X_t`.\n"
            "- `quintic_sv/pricing.py` pour le pricing SPX et VIX.\n"
            "- `quintic_sv/configs.py` pour les parametres des figures du papier.\n"
            "- `quintic_sv/market.py` pour simuler un pseudo-marche en absence de donnees reelles.\n"
            "- `quintic_sv/plots.py` pour tracer les figures."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Donnees externes manquantes\n\n"
            "Le papier a besoin de trois sources externes :\n\n"
            "- options SPX par maturite et strike ;\n"
            "- options VIX par maturite et strike ;\n"
            "- futures VIX pour la courbe a terme.\n\n"
            "Sans ces donnees, on fait deux choses :\n\n"
            "- on garde les parametres publies dans le papier pour reproduire les courbes du modele ;\n"
            "- on simule des bid/ask et une courbe `xi_0` de marche autour des courbes du modele."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if not (ROOT / 'quintic_sv').exists():\n"
            "    ROOT = ROOT.parent\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from quintic_sv.paper_figures import (\n"
            "    build_figure_1_bundle,\n"
            "    build_figure_3_bundle,\n"
            "    build_figure_4_bundle,\n"
            "    build_figure_5_bundle,\n"
            ")\n"
            "from quintic_sv.plots import plot_forward_curve_comparison, plot_joint_smiles, plot_time_dependent_h\n\n"
            "# 1.0 = budget proche du papier ; plus petit = execution plus rapide.\n"
            "budget_scale = 0.35"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Modele de volatilite\n\n"
            "Formule utilisee :\n\n"
            "`dS_t / S_t = sigma_t dB_t` avec `sigma_t = sqrt(xi_0(t)) p(X_t) / sqrt(E[p(X_t)^2])` et `p(x) = alpha_0 + alpha_1 x + alpha_3 x^3 + alpha_5 x^5`.\n\n"
            "Entrees : `rho`, `H`, `eps`, les coefficients `alpha`, et la courbe `xi_0`.\n\n"
            "Sortie : une volatilite instantanee `sigma_t` et donc une dynamique pour `S_t`.\n\n"
            "Utilite : c'est le coeur du modele, commun au pricing SPX et VIX."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Pricing SPX/VIX avec `H` constant et courbe spline\n\n"
            "Formules utilisees :\n\n"
            "- SPX : Monte Carlo exact sur `X_t` puis turbocharging avec antithetiques et control variate.\n"
            "- VIX : `VIX_T^2 = (100^2 / Delta) * integral_T^(T+Delta) xi_T(u) du`, qui devient un polynome en `X_T`.\n\n"
            "Entrees : les parametres calibres de la Figure 1, plus la courbe `xi_0` spline.\n\n"
            "Sorties : smiles SPX, smiles VIX, et courbe de variance forward.\n\n"
            "Utilite : reproduire les Figures 1 et 2."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "bundle_12 = build_figure_1_bundle(budget_scale=budget_scale)\n"
            "fig_1 = plot_joint_smiles(bundle_12['spx_smiles'], bundle_12['vix_smiles'], orientation='side_by_side')\n"
            "fig_2 = plot_forward_curve_comparison(bundle_12['forward_curve'])\n"
            "fig_1\n"
            "fig_2"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Courbe de variance parametrique\n\n"
            "Formule utilisee : `xi_0(t) = a e^{-b t} + c (1 - e^{-b t})`.\n\n"
            "Entrees : `a`, `b`, `c` et les memes parametres du modele pour `X_t`.\n\n"
            "Sortie : une courbe `xi_0(t)` plus rigide que la spline, mais simple a calibrer.\n\n"
            "Utilite : le papier l'utilise quand on ajuste peu de slices, ici pour les Figures 3 et 4."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "bundle_3 = build_figure_3_bundle(budget_scale=budget_scale)\n"
            "bundle_4 = build_figure_4_bundle(budget_scale=budget_scale)\n"
            "fig_3 = plot_joint_smiles(bundle_3['spx_smiles'], bundle_3['vix_smiles'], orientation='side_by_side')\n"
            "fig_4 = plot_joint_smiles(bundle_4['spx_smiles'], bundle_4['vix_smiles'], orientation='side_by_side')\n"
            "fig_3\n"
            "fig_4"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. `H(t)` dependant du temps\n\n"
            "Formule utilisee : `H(t) = H_0 e^{-kappa t} + H_inf (1 - e^{-kappa t})`.\n\n"
            "Entrees : `H_0`, `H_inf`, `kappa`, `eps`, les coefficients `alpha`, et une courbe spline `xi_0`.\n\n"
            "Sorties : smiles plus longs en maturite, courbe de variance forward, et courbe `H(t)`.\n\n"
            "Utilite : ameliorer les fits au-dela de 3-4 mois, comme dans les Figures 5 a 7."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "bundle_567 = build_figure_5_bundle(budget_scale=budget_scale)\n"
            "fig_5 = plot_joint_smiles(bundle_567['spx_smiles'], bundle_567['vix_smiles'], orientation='stacked')\n"
            "fig_6 = plot_forward_curve_comparison(bundle_567['forward_curve'])\n"
            "fig_7 = plot_time_dependent_h(bundle_567['h_times'], bundle_567['h_values'])\n"
            "fig_5\n"
            "fig_6\n"
            "fig_7"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Ou brancher de vraies donnees plus tard\n\n"
            "Si tu recuperes des donnees CBOE, les points de branchement sont :\n\n"
            "- dans `quintic_sv/market.py` pour remplacer les bid/ask simules ;\n"
            "- dans `quintic_sv/configs.py` pour injecter les vraies maturites et courbes ;\n"
            "- juste avant la construction de `xi_0`, en remplacant la courbe synthetique par une courbe extraite via Carr-Madan.\n\n"
            "Dans cette version, le notebook reste volontairement bref et montre le bon endroit logique pour chaque entree/sortie."
        )
    )

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.13"}
    return nb


def main() -> None:
    output_dir = ensure_directory(ROOT / "notebooks")
    notebook = build_notebook()
    destination = output_dir / "quintic_sv_reproduction.ipynb"
    nbf.write(notebook, destination)
    print(f"Notebook written to: {destination}")


if __name__ == "__main__":
    main()
