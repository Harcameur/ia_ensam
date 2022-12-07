""" ### MODULE DECOUVERTE DE LA BASE
"""
import matplotlib.pyplot as plt
import numpy as np

from data.data2 import medic

NUMBER_OF_PATH = 6

cx, pathologie = medic()


def histogramme(_list : np.ndarray, title="", xlabel="", bins=None):
    """Affichage de l'histogramme

    Args:
        _list (np.ndarray): list des données
        title (str, optional): Titre de l'histogramme. Defaults to "".
        xlabel (str, optional): Label de l'axe X. Defaults to "".
        bins (int, optional): nombre de barre dans l'histogramme. 
            Defaults to None.
    """
    plt.hist(
        _list, bins, density=True, facecolor='g', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel('Densité')
    plt.title(title)
    plt.grid(True)
    plt.show()


def rendu_courbes():
    """Affichages des courbes demandés par le tp
    """
    histogramme(
        pathologie, title="Histogrammes des apparitions des pathologies",
        xlabel="Pathologie", bins=6)
    histogramme(
        cx[:, 0], title="Distributions des ages",
        xlabel="Age", bins=50)


if __name__ == "__main__":
    rendu_courbes()
