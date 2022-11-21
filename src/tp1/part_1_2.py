from cours.data import circle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier


""" ### PARTIE 1 CHARGEMENT ET AFFICHAGE DES DONNEES
"""

valeurs, classe = circle()


def carre(x):
    return [x[0]**2, x[1]**2]


# Valeurs carré :
# valeurs = np.array(list(map(carre, valeurs)))

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

mycolormap = plt.cm.PiYG  # carte de couleur

# points de couleur des données
plt.scatter(
    valeurs[:, 0], valeurs[:, 1],
    c=classe, cmap=mycolormap, edgecolors='black')


""" ### PARTIE 2 UN PREMIER RESEAU
"""

mlp = MLPClassifier(
    hidden_layer_sizes=(50,), activation='tanh', solver='lbfgs')
mlp.fit(valeurs, classe)  # apprentissage

# utilisation du réseau entrainé pour créer le fond du graphique
out = mlp.predict_proba(np.asarray(image))
Z = out[:, 1]  # on récupère les valeurs pour construire le fond
Z = Z.reshape((len(ys), len(xs)))  # mise en forme des valeurs du fond
plt.contourf(xs, ys, Z, cmap=mycolormap, alpha=.6)  # fond de l'affichage

plt.show()  # affichage
