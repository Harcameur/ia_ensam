import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# chargement des données, le fichier data.py doit être dans le même dossier
valeurs, classe = data.test()

# la variable valeurs est de la forme [[x0, y0], [x1, y1], [x2, y2] ...]
# la variable classe est de la forme  [c0, c1, c2, ...]
# avec la relation entrée -> sortie : [x0, y0] -> c0, [x1, y1] -> c1, ...

mlp = MLPClassifier(hidden_layer_sizes=(1,), solver='lbfgs') # création du réseau à un seul neurone

mlp.fit(valeurs, classe) # apprentissage

# à partir d'ici l'entraînement du réseau est terminé, on peut utiliser le réseau entraîné de la façon suivante :
# pour prédire les classes aux coordonnées [0, 0] et [5, 5] on peut écrire :
# classes = mlp.predict([[0, 0], [5, 5]])

# création d'un graphique

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
        image.append([x,y])

out = mlp.predict_proba(np.asarray(image)) # utilisation du réseau entrainé pour créer le fond du graphique
Z = out[:, 1] # on récupère les valeurs pour construire le fond
Z = Z.reshape((len(ys), len(xs))) # mise en forme des valeurs du fond

mycolormap = plt.cm.bwr # carte de couleur bwr (bleu -> blanc -> rouge)

plt.contourf(xs, ys, Z, cmap=mycolormap, alpha=.6) # fond de l'affichage
plt.scatter(valeurs[:,0], valeurs[:,1], c=classe, cmap=mycolormap, edgecolors='black') # points de couleur des données
plt.show() # affichage
