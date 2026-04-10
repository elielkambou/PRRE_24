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
            "# Reproduction du modele Quintic SV, version 2 detaillee\n\n"
            "Cette version montre les calculs intermediaires. La premiere version du notebook reste intacte ; ici on ouvre la boite noire pour voir comment les courbes sont fabriquees."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Ce qu'on va faire\n\n"
            "On suit le cas `H` constant avec courbe spline, c'est celui des Figures 1 et 2 du papier. Puis on refait le calcul VIX plus explicitement.\n\n"
            "Idee generale :\n\n"
            "1. simuler le facteur gaussien `X_t` ;\n"
            "2. construire le polynome `p(X_t)` ;\n"
            "3. construire la volatilite instantanee `sigma_t` ;\n"
            "4. pricer les calls SPX ;\n"
            "5. pricer les calls VIX ;\n"
            "6. tracer les courbes, puis superposer un pseudo-marche simule."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if not (ROOT / 'quintic_sv').exists():\n"
            "    ROOT = ROOT.parent\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from quintic_sv.configs import paper_constant_h_spline_scenario\n"
            "from quintic_sv.curves import evaluate_spline_forward_curve\n"
            "from quintic_sv.market import add_synthetic_market_to_spx, add_synthetic_market_to_vix, simulate_forward_curve_comparison\n"
            "from quintic_sv.model import normalization_variance, simulate_xt_grid_constant_h\n"
            "from quintic_sv.plots import plot_forward_curve_comparison, plot_joint_smiles\n"
            "from quintic_sv.pricing import (\n"
            "    _control_variate_prices,\n"
            "    _gauss_legendre,\n"
            "    _gaussian_density,\n"
            "    _integrated_beta_polynomial,\n"
            "    price_spx_smile_constant_h_spline,\n"
            ")\n"
            "from quintic_sv.black import implied_volatility_vector\n"
            "from quintic_sv.utils import VIX_WINDOW_YEARS, generate_antithetic_normals, horner_vector, spx_log_moneyness_range\n\n"
            "plt.rcParams['figure.figsize'] = (8, 3)\n"
            "scenario = paper_constant_h_spline_scenario()\n"
            "coeffs = scenario.params.coefficients()\n"
            "scenario"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Parametres d'entree\n\n"
            "Formule utilisee :\n\n"
            "$$\\sigma_t = \\sqrt{\\xi_0(t)} \\frac{p(X_t)}{\\sqrt{\\mathbb E[p(X_t)^2]}}$$\n\n"
            "avec\n\n"
            "$$p(x)=\\alpha_0 + \\alpha_1 x + \\alpha_3 x^3 + \\alpha_5 x^5$$\n\n"
            "Entrees ici : `rho`, `H`, `eps`, les coefficients `alpha`, et les noeuds de la courbe `xi_0`.\n\n"
            "Sortie plus tard : une trajectoire de volatilite `sigma_t`."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "spot = scenario.spot\n"
            "maturity = float(scenario.spx_maturities[1])  # 30 jours environ\n"
            "n_steps = 120\n"
            "n_sims = 1200\n"
            "normals = generate_antithetic_normals(n_steps, n_sims, scenario.seed)\n\n"
            "print('spot =', spot)\n"
            "print('maturity =', maturity)\n"
            "print('rho =', scenario.params.rho)\n"
            "print('H =', scenario.params.H)\n"
            "print('eps =', scenario.params.eps)\n"
            "print('coeffs =', coeffs)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Simulation du facteur `X_t`\n\n"
            "Formule utilisee :\n\n"
            "$$dX_t = -(1/2-H)\\varepsilon^{-1} X_t dt + \\varepsilon^{H-1/2} dW_t$$\n\n"
            "Entrees : `H`, `eps`, la maturite et des gaussiennes standard.\n\n"
            "Sorties : la grille de temps, les trajectoires `X_t`, et l'ecart-type theorique de `X_t` a chaque date.\n\n"
            "Utilite : `X_t` est le seul facteur qui pilote toute la volatilite."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "times, xt_paths, std_x = simulate_xt_grid_constant_h(scenario.params.H, scenario.params.eps, maturity, normals)\n\n"
            "print('shape times   =', times.shape)\n"
            "print('shape X paths =', xt_paths.shape)\n"
            "print('shape std_x   =', std_x.shape)\n"
            "print('premieres valeurs std_x =', np.round(std_x[:8], 6))\n"
            "print('extrait trajectoire X_t =', np.round(xt_paths[:8, 0], 6))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(9, 3))\n"
            "for idx in range(6):\n"
            "    plt.plot(times, xt_paths[:, idx], linewidth=0.9)\n"
            "plt.title('Exemples de trajectoires du facteur X_t')\n"
            "plt.xlabel('time')\n"
            "plt.ylabel('X_t')\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Calcul du polynome `p(X_t)` et de la normalisation\n\n"
            "Formule utilisee :\n\n"
            "$$p(X_t)=\\alpha_0 + \\alpha_1 X_t + \\alpha_3 X_t^3 + \\alpha_5 X_t^5$$\n\n"
            "et\n\n"
            "$$\\mathbb E[p(X_t)^2]$$\n\n"
            "Entrees : les coefficients `alpha` et les trajectoires `X_t`.\n\n"
            "Sorties : la valeur brute `p(X_t)` et la normalisation theorique.\n\n"
            "Utilite : cette normalisation permet a `xi_0(t)` de jouer le role de courbe de variance forward."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "p_xt = horner_vector(coeffs[::-1], xt_paths)\n"
            "normal_var = normalization_variance(coeffs, std_x)\n\n"
            "print('shape p(X_t) =', p_xt.shape)\n"
            "print('shape E[p(X_t)^2] =', normal_var.shape)\n"
            "print('extrait p(X_t) =', np.round(p_xt[:8, 0], 6))\n"
            "print('extrait E[p(X_t)^2] =', np.round(normal_var[:8], 6))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Construction de la courbe `xi_0(t)` et de la volatilite `sigma_t`\n\n"
            "Formule utilisee :\n\n"
            "$$\\sigma_t = \\sqrt{\\xi_0(t)} \\frac{p(X_t)}{\\sqrt{\\mathbb E[p(X_t)^2]}}$$\n\n"
            "Entrees : les noeuds spline de `xi_0`, `p(X_t)` et `E[p(X_t)^2]`.\n\n"
            "Sortie : une trajectoire de volatilite instantanee pour chaque simulation.\n\n"
            "Utilite : c'est l'objet directement injecte dans le pricing SPX."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "forward_curve = evaluate_spline_forward_curve(times, scenario.forward_node_times, scenario.forward_node_values)\n"
            "volatility = np.sqrt(forward_curve)[:, None] * p_xt / np.sqrt(normal_var)[:, None]\n\n"
            "print('shape xi_0(t) =', forward_curve.shape)\n"
            "print('shape sigma_t =', volatility.shape)\n"
            "print('extrait xi_0(t) =', np.round(forward_curve[:8], 6))\n"
            "print('extrait sigma_t =', np.round(volatility[:8, 0], 6))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(9, 3))\n"
            "plt.plot(times, forward_curve, label='xi_0(t)', linewidth=1.5)\n"
            "plt.plot(times, volatility[:, 0], label='sigma_t, path 1', linewidth=1.0)\n"
            "plt.title('Courbe forward et volatilite instantanee')\n"
            "plt.xlabel('time')\n"
            "plt.legend()\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Pricing SPX\n\n"
            "Le papier utilise Monte Carlo pour SPX, avec antithetiques et control variate.\n\n"
            "Entrees : `sigma_t`, `rho`, `S_0`, les strikes, la maturite et les gaussiennes.\n\n"
            "Sorties : prix de calls SPX, puis volatilites implicites.\n\n"
            "Utilite : produire les smiles SPX du papier."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "lm_min, lm_max = spx_log_moneyness_range(maturity)\n"
            "spx_strikes = np.exp(np.linspace(lm_min, lm_max, 20)) * spot\n\n"
            "spx_prices, spx_conf_scale = _control_variate_prices(\n"
            "    scenario.params.rho,\n"
            "    maturity,\n"
            "    spot,\n"
            "    spx_strikes,\n"
            "    volatility,\n"
            "    normals,\n"
            "    n_sims,\n"
            ")\n"
            "spx_iv = implied_volatility_vector(spx_prices, spot, spx_strikes, maturity)\n\n"
            "print('extrait prix calls SPX =', np.round(spx_prices[:6], 6))\n"
            "print('extrait IV SPX =', np.round(spx_iv[:6], 6))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(8, 3))\n"
            "plt.plot(np.log(spx_strikes / spot), spx_iv, color='green')\n"
            "plt.title('Smile SPX calculee explicitement')\n"
            "plt.xlabel(r'log moneyness $\\log(K/S_0)$')\n"
            "plt.ylabel('implied vol')\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 6. Pricing VIX\n\n"
            "Formule utilisee :\n\n"
            "$$\\mathrm{VIX}_T^2 = \\frac{100^2}{\\Delta} \\int_T^{T+\\Delta} \\xi_T(u) \\, du$$\n\n"
            "Dans ce modele, `VIX_T^2` devient un polynome en `X_T` avec des coefficients `beta_i`.\n\n"
            "Entrees : `H`, `eps`, `alpha`, la courbe `xi_0`, la maturite `T` et la fenetre VIX `Delta = 30/360`.\n\n"
            "Sorties : future VIX, prix des calls VIX, puis volatilites implicites VIX.\n\n"
            "Utilite : produire les smiles VIX du papier."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "vix_maturity = float(scenario.vix_maturities[1])  # environ 23 jours\n"
            "delta = VIX_WINDOW_YEARS\n"
            "u_grid = np.linspace(vix_maturity, vix_maturity + delta, scenario.vix_n_steps + 1)\n"
            "dt = delta / scenario.vix_n_steps\n\n"
            "kappa_tilde = (0.5 - scenario.params.H) / scenario.params.eps\n"
            "eta_tilde = scenario.params.eps ** (scenario.params.H - 0.5)\n"
            "exp_det = np.exp(-kappa_tilde * (u_grid - vix_maturity))\n"
            "std_g = eta_tilde * np.sqrt(1.0 / (2.0 * kappa_tilde) * (1.0 - np.exp(-2.0 * kappa_tilde * (u_grid - vix_maturity))))\n"
            "std_x_u = eta_tilde * np.sqrt(1.0 / (2.0 * kappa_tilde) * (1.0 - np.exp(-2.0 * kappa_tilde * u_grid)))\n"
            "std_x_t = eta_tilde * np.sqrt(1.0 / (2.0 * kappa_tilde) * (1.0 - np.exp(-2.0 * kappa_tilde * vix_maturity)))\n\n"
            "cauchy_product = np.convolve(coeffs, coeffs)\n"
            "normal_var_u = normalization_variance(coeffs, std_x_u)\n"
            "forward_curve_u = evaluate_spline_forward_curve(u_grid, scenario.forward_node_times, scenario.forward_node_values)\n"
            "beta_integrated = _integrated_beta_polynomial(cauchy_product, exp_det, std_g, forward_curve_u, normal_var_u, dt)\n\n"
            "print('extrait beta_i =', np.round(beta_integrated[:6], 6))\n"
            "print('std_X_T =', round(float(std_x_t), 6))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "y_nodes, y_weights = _gauss_legendre(-8.0, 8.0, 250)\n"
            "density = _gaussian_density(y_nodes)\n"
            "vix_values = np.sqrt(np.maximum(horner_vector(beta_integrated[::-1], std_x_t * y_nodes) / delta, 0.0))\n"
            "vix_future = np.sum(density * vix_values * y_weights)\n"
            "vix_strike_perc = np.exp(np.linspace(-0.1, 1.0, 25))\n"
            "vix_strikes = vix_future * vix_strike_perc\n"
            "vix_prices = np.sum(density[None, :] * np.maximum(vix_values[None, :] - vix_strikes[:, None], 0.0) * y_weights[None, :], axis=1)\n"
            "vix_iv = implied_volatility_vector(vix_prices, vix_future, vix_strikes, vix_maturity)\n\n"
            "print('future VIX =', round(float(100.0 * vix_future), 6))\n"
            "print('extrait prix calls VIX =', np.round(100.0 * vix_prices[:6], 6))\n"
            "print('extrait IV VIX =', np.round(vix_iv[:6], 6))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(8, 3))\n"
            "plt.plot(100.0 * vix_strikes, vix_iv, color='green')\n"
            "plt.axvline(100.0 * vix_future, color='black', linewidth=0.8)\n"
            "plt.title('Smile VIX calculee explicitement')\n"
            "plt.xlabel('strike')\n"
            "plt.ylabel('implied vol')\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 7. Passage aux courbes finales du papier\n\n"
            "Ici on repasse par les fonctions haut niveau du projet. Elles font exactement les memes types de calculs, mais pour toutes les maturites d'une figure.\n\n"
            "Ensuite on ajoute un pseudo-marche simule : points bleu/rouge pour imiter bid/ask, et une courbe `xi_0` de marche synthetique pour illustrer la partie \"donnees externes\" absente."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "smile_full = price_spx_smile_constant_h_spline(\n"
            "    scenario.params.rho,\n"
            "    scenario.params.H,\n"
            "    scenario.params.eps,\n"
            "    maturity,\n"
            "    coeffs,\n"
            "    spot,\n"
            "    spx_strikes,\n"
            "    scenario.forward_node_times,\n"
            "    scenario.forward_node_values,\n"
            "    normals,\n"
            "    n_sims,\n"
            ")\n"
            "smile_full = add_synthetic_market_to_spx(smile_full, scenario.market_seed)\n"
            "curve_compare = simulate_forward_curve_comparison(scenario.forward_node_times, scenario.forward_node_values, scenario.market_seed + 1)\n\n"
            "print('shape smile_full model_iv =', smile_full.model_iv.shape)\n"
            "print('shape pseudo market bid   =', smile_full.market_bid_iv.shape)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from quintic_sv.pricing import price_vix_smile_constant_h_spline\n\n"
            "vix_smile_full = price_vix_smile_constant_h_spline(\n"
            "    scenario.params.H,\n"
            "    scenario.params.eps,\n"
            "    vix_maturity,\n"
            "    coeffs,\n"
            "    vix_strike_perc,\n"
            "    scenario.forward_node_times,\n"
            "    scenario.forward_node_values,\n"
            "    quadrature_degree=250,\n"
            "    n_steps=scenario.vix_n_steps,\n"
            ")\n"
            "vix_smile_full = add_synthetic_market_to_vix(vix_smile_full, scenario.market_seed + 2)\n\n"
            "fig_joint = plot_joint_smiles([smile_full], [vix_smile_full], orientation='side_by_side')\n"
            "fig_curve = plot_forward_curve_comparison(curve_compare)\n"
            "fig_joint\n"
            "fig_curve"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## A retenir\n\n"
            "- le notebook V1 montrait bien les vraies courbes, mais en mode pipeline ;\n"
            "- ce notebook V2 montre les objets intermediaires qui fabriquent ces courbes ;\n"
            "- les lignes vertes viennent du vrai calcul du modele ;\n"
            "- les points bleu/rouge et la courbe \"market\" sont simules ici parce qu'on n'a pas les donnees CBOE."
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
    destination = output_dir / "quintic_sv_reproduction_v2_detailed.ipynb"
    nbf.write(notebook, destination)
    print(f"Notebook written to: {destination}")


if __name__ == "__main__":
    main()
