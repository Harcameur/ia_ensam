""" ### PARTIE 3 QUELLE QUANTITE DE DONNEES UTILISER?
"""
from cours.data import cross
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


valeurs, classe = cross()

train_size = 10

valeursTrain, valeursTest, classeTrain, classeTest =\
    train_test_split(
        valeurs, classe, stratify=classe,
        train_size=train_size, test_size=0.3)

x_min = valeursTrain[:, 0].min() - .5
x_max = valeursTrain[:, 0].max() + .5
y_min = valeursTrain[:, 1].min() - .5
y_max = valeursTrain[:, 1].max() + .5

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
    valeursTrain[:, 0], valeursTrain[:, 1],
    c=classeTrain, cmap=mycolormap, edgecolors='black')

plt.scatter(
    valeursTest[:, 0], valeursTest[:, 1],
    c=classeTest, cmap=mycolormap, edgecolors='black', marker="o")

mlp = MLPClassifier(
    hidden_layer_sizes=(50,), activation='tanh', solver='lbfgs')
mlp.fit(valeursTrain, classeTrain)  # apprentissage

# utilisation du réseau entrainé pour créer le fond du graphique
out = mlp.predict_proba(np.asarray(image))
Z = out[:, 1]  # on récupère les valeurs pour construire le fond
Z = Z.reshape((len(ys), len(xs)))  # mise en forme des valeurs du fond

score = mlp.score(valeursTest, classeTest)
plt.title(
    f"train_size = {str(train_size)}, test_size = 33%, score = {score}")
plt.contourf(xs, ys, Z, cmap=mycolormap, alpha=.6)  # fond de l'affichage

print(score)

plt.show()  # affichage
