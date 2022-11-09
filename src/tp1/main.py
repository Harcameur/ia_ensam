from cours.data import circle
import matplotlib.pyplot as plt
import numpy as np

""" ### PARTIE 1 CHARGEMENT ET AFFICHAGE DES DONNEES
"""

valeurs, classe = circle()

# bornes min/max du graphique
x_min = valeurs[:, 0].min() - .5
x_max = valeurs[:, 0].max() + .5
y_min = valeurs[:, 1].min() - .5
y_max = valeurs[:, 1].max() + .5

# création de la couleur de fond
image = []
xs = np.arange(x_min, x_max, 0.05)
ys = np.arange(y_min, y_max, 0.05)
for y in ys:
    for x in xs:
        image.append([x, y])

mycolormap = plt.cm.Spectral  # carte de couleur

# points de couleur des données
plt.scatter(
    valeurs[:, 0], valeurs[:, 1],
    c=classe, cmap=mycolormap, edgecolors='black')
plt.show()  # affichage
