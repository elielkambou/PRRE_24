# Quintic SV Reproduction

Ce projet reconstruit la methode du papier `Quintic_SV_model.pdf` avec une architecture lisible et honnete :

- les parametres publies dans le papier sont traites comme des entrees ;
- le calcul des smiles et des courbes est effectue explicitement par le code ;
- la partie SPX dispose de deux moteurs : Monte Carlo de reference et surrogate deep learning ;
- le notebook principal montre clairement quelles fonctions sont appelees et dans quels fichiers elles vivent.

## Ce qui a ete volontairement retire

Le depot ne contient plus :

- de pseudo-marche invente ;
- de courbes "market" synthetiques ;
- de notebook genere par script ;
- d'artefacts PNG ou de caches Python versionnes.

Le seul document original conserve tel quel est `Quintic_SV_model.pdf`.

## Point d'entree

Le point d'entree principal est :

- `notebooks/quintic_sv_reproduction.ipynb`

Ce notebook montre :

- les inputs publies dans le papier ;
- la simulation du facteur gaussien ;
- la construction de la volatilite quintique ;
- le pricing VIX analytique ;
- le pricing SPX Monte Carlo ;
- l'entrainement du surrogate deep learning qui remplace le Monte Carlo SPX ;
- la reproduction honnete des figures du papier a partir des inputs publies.

## Organisation du code

```text
.
|- Quintic_SV_model.pdf
|- README.md
|- requirements.txt
|- notebooks/
|  `- quintic_sv_reproduction.ipynb
|- scripts/
|  |- generate_paper_figures.py
|  `- train_spx_surrogate.py
`- quintic_sv/
   |- __init__.py
   |- black.py
   |- configs.py
   |- curves.py
   |- factor_process.py
   |- plots.py
   |- polynomial_volatility.py
   |- spx_deep_learning.py
   |- spx_monte_carlo.py
   |- paper_workflow.py
   |- types.py
   |- utils.py
   `- vix_analytic.py
```

## Workflow scientifique

1. `configs.py` fournit les inputs publies dans le papier.
2. `factor_process.py` simule le facteur gaussien `X_t`.
3. `polynomial_volatility.py` construit `p(X_t)` puis `sigma_t`.
4. `vix_analytic.py` price les smiles VIX par integration gaussienne.
5. `spx_monte_carlo.py` price les smiles SPX par Monte Carlo de reference.
6. `spx_deep_learning.py` entraine un MLP Numpy sur des labels Monte Carlo puis remplace le moteur SPX.
7. `paper_workflow.py` assemble les bundles de reproduction des figures.

## Installation rapide

```bash
python3 -m pip install -r requirements.txt
```

## Usage rapide

Entrainer un surrogate SPX :

```bash
python3 scripts/train_spx_surrogate.py --output outputs/models/spx_surrogate_demo.npz
```

Generer les figures du papier avec le moteur Monte Carlo :

```bash
python3 scripts/generate_paper_figures.py --spx-engine mc
```

Generer les figures du papier avec le surrogate deep learning :

```bash
python3 scripts/generate_paper_figures.py --spx-engine dl --surrogate-model outputs/models/spx_surrogate_demo.npz
```

## Important

Le papier decrit la fonction objectif et la logique generale de calibration, mais il ne fournit pas tout le pipeline numerique detaille. Ici, on reproduit donc les sorties du papier a partir des inputs publies, sans pretendre refaire la calibration complete sur donnees CBOE.
