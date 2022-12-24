"""MODULE ETUDE AVANCEE
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from data.data2 import medic
from part2 import mlp_apprentissage, color_score


cx, pathologie = medic()
NOMBRE_CARACTERISTIQUE = 8
TEST_RATIO = .4


def filtre_cx(_list_id: list) -> np.ndarray:
    """filtre des données d'entrées en supprimant le(s) identifiant(s) de
    colonne

    Args:
        _list_id (list): listes des ids de la colonne, ici de 0 à 7, à
        supprimer

    Returns:
        np.ndarray: tabelau filtré
    """
    return np.delete(cx, _list_id, 1)


def creation_sets(_list_id: list) -> tuple:
    """Creation de jeu de donnée en fonction de la liste des caractéritiques
    filtrés

    Args:
        _list_id (list): list des identifiants à filtrer

    Returns:
        tuple: cx_train, cx_test, pathologie_train, pathologie_test
    """
    cx_filtred = filtre_cx(_list_id)
    cx_train, cx_test, pathologie_train, pathologie_test =\
        train_test_split(
            cx_filtred, pathologie, test_size=TEST_RATIO)
    return cx_train, cx_test, pathologie_train, pathologie_test


def show_score(
        _mlp: MLPClassifier,
        cx_train: np.ndarray,
        cx_test: np.ndarray,
        pathologie_train: np.ndarray,
        pathologie_test: np.ndarray) -> None:
    """Affiche dans la console le score d'entrainemnet et de test selon le 
    filtrage choisi en amont

    Args:
        _mlp (MLPClassifier): le classifer entrainé
        cx_train (np.ndarray): le tableau d'entré filtré
        cx_test (np.ndarray): le tableau de test aussi filtré par extension
        pathologie_train (np.ndarray): tableau des pathologies d'entrainement
        pathologie_test (np.ndarray): tableau des pathologies de test
    """
    _score_train = _mlp.score(cx_train, pathologie_train)
    _score_test = _mlp.score(cx_test, pathologie_test)
    print(f"Score d'entrainement : {color_score(_score_train)}")
    print(f"Score de test : {color_score(_score_test)}")


def training_enlevant_une_caracteristique() -> None:
    """Comparaison des scores en enlevant 1 caractéritique
    """
    print("Démarrage des entrainements")
    for _id in range(NOMBRE_CARACTERISTIQUE):
        print(
            f"\033[94m\033[1m Demarrage Sans la caractéritique : \033[4m{_id}\
\033[0m")
        cx_train, cx_test, pathologie_train, pathologie_test =\
            creation_sets([_id])
        _mlp = mlp_apprentissage(cx_train, pathologie_train)
        show_score(_mlp, cx_train, cx_test, pathologie_train, pathologie_test)


if __name__ == "__main__":
    # Question 1: Caractéristique par caractéristique
    training_enlevant_une_caracteristique()
    # Question 2: Caractéristique comibné
