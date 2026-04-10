# Quintic SV Reproduction

Ce projet reprend le papier `Quintic_SV_model.pdf` et le notebook public des auteurs pour reconstruire les principales courbes du modele de volatilite quintique d'Ornstein-Uhlenbeck.

## Structure proposee

```text
.
|- Quintic_SV_model.pdf
|- authors_notebook.ipynb
|- requirements.txt
|- data/
|  |- README.md
|  |- raw/
|  `- simulated/
|- notebooks/
|  `- quintic_sv_reproduction.ipynb
|- outputs/
|  `- figures/
|- scripts/
|  |- build_notebook.py
|  `- generate_paper_figures.py
`- quintic_sv/
   |- __init__.py
   |- black.py
   |- configs.py
   |- curves.py
   |- market.py
   |- model.py
   |- paper_figures.py
   |- plots.py
   |- pricing.py
   |- types.py
   `- utils.py
```

## Ce qui vient du papier

- Les formules du modele et du pricing VIX/SPX.
- Les jeux de parametres publies dans les Figures 1 a 7.
- La logique Monte Carlo avec antithetiques et control variate.
- La formule de `H(t)` dependant du temps.

## Ou il faut normalement des donnees externes

Le papier calibre le modele sur des donnees CBOE. En pratique, il faut :

- la surface d'options SPX par maturite et strike ;
- la surface d'options VIX par maturite et strike ;
- la courbe des futures VIX ;
- une interpolation lisse du smile SPX pour extraire `xi_0` via la formule de Carr-Madan.

Comme ces donnees ne sont pas disponibles ici, le projet :

- reutilise les parametres publies dans le papier pour reproduire les courbes du modele ;
- simule des bid/ask "pseudo-marche" autour des courbes du modele ;
- simule une courbe de variance forward "stripped market" autour de la courbe calibree.

## Lancer le projet

```powershell
python -m pip install -r requirements.txt
python scripts\generate_paper_figures.py
python scripts\build_notebook.py
```

Les figures seront ecrites dans `outputs/figures/` et le notebook dans `notebooks/`.
