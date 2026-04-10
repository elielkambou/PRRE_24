# Donnees attendues

## Si on avait les vraies donnees

Pour reproduire une calibration reelle comme dans le papier, il faudrait deposer ici :

- des cotations d'options SPX par maturite/strike ;
- des cotations d'options VIX par maturite/strike ;
- la courbe des futures VIX ;
- eventuellement une surface SPX deja lissee par SVI ou SABR.

## Dans cette version

Le projet ne dispose pas de donnees CBOE. On simule donc :

- des bid/ask autour des volatilites implicites generees par le modele ;
- une courbe `xi_0` "marche" synthetique autour de la courbe calibree du papier ;
- des niveaux de futures VIX synthetiques proches du modele.
