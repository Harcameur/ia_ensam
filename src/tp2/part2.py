""" MODULE APPRENTISSAGE SUR LA BASE ET ETUDE AVANCEE
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


from data.data2 import medic

NUMBER_OF_PATH = 6
cx, pathologie = medic()
TAILLE_TABLEAU = len(cx)
TEST_RATIO = .4


def creating_sets(_train_size: int | None) -> tuple:
    """Créer un set de données selon la train_size

    Args:
        _train_size (int | None): taille des jeu de données d'entrainement

    Returns:
        tuple: 4 list cx_train, cx_test, pathologie_train, pathologie_test
    """
    cx_train, cx_test, pathologie_train, pathologie_test =\
        train_test_split(
            cx, pathologie, train_size=_train_size, test_size=TEST_RATIO)
    return cx_train, cx_test, pathologie_train, pathologie_test


def mlp_apprentissage(
        _cx_train: list, _pathologie_train: list) -> MLPClassifier:
    """Fonction d'apprentissage

    Args:
        _cx_train (list): donnée d'entré d'entrainement
        _pathologie_train (list): donnée de sortie d'entrainement

    Returns:
        MLPClassifier: objet IA
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(50, 6),
        activation='tanh',
        solver='adam',
        max_iter=1000)

    mlp.fit(_cx_train, _pathologie_train)  # apprentissage
    return mlp


def filter_cx_by_path(
        _cx_test: list, _pathologie_test: list, path_num: int) -> tuple:
    """Filtre les données pour quelles correspondent à une pathologie

    Args:
        _cx_test (list): jeu de test d'entrée à filtrer
        _pathologie_test (list): jeu de test de sortie à filtrer
        path_num (int): numéro de pathologie

    Returns:
        tuple: les deux jeux de données filtrés
    """
    _filtered_cx = [
        _cx_test[i]
        for i in range(len(_cx_test))
        if _pathologie_test[i] == path_num]
    _filtered_pathologie_test =\
        list(filter(lambda x: x == path_num, _pathologie_test))
    return _filtered_cx, _filtered_pathologie_test


def train_size_score_variation(_min: int, qte: int) -> list:
    """Creation de liste avec des valeurs de taille croissante (train_size)
    de jeu de données

    Args:
        _min (int): taille minimum
        qte (int): nombre de valeurs

    Returns:
        list: listes des tailles
    """
    _list_train_size = np.linspace(
        _min, int(TAILLE_TABLEAU*(1-TEST_RATIO)), qte)

    _entier_list_train_size = list(map(lambda x: int(x), _list_train_size))

    return _entier_list_train_size


def get_score_for_spec_train_size(_train_size: int) -> tuple:
    """Récupération des scores pour une taille de donnée fixés

    Args:
        _train_size (int): taille du jeu

    Returns:
        tuple: score d'entrainement, score de test
    """
    _cx_train, _cx_test, _pathologie_train, _pathologie_test =\
        creating_sets(_train_size)
    _mlp = mlp_apprentissage(_cx_train, _pathologie_train)
    return _mlp.score(_cx_train, _pathologie_train),\
        _mlp.score(_cx_test, _pathologie_test)


def comparaison_train_size() -> None:
    """Fonction principale pour la question 1 de comparaison de score selon le
    jeu de données
    """
    print("Attention cette partie peut prendre du temps")
    _entier_list_train_size = train_size_score_variation(100, 20)
    print(_entier_list_train_size, TAILLE_TABLEAU)
    _train_score_list = []
    _test_score_list = []
    for _train_size in _entier_list_train_size:
        print(
            f"\033[94m\033[1m- Démarrage du train_size = \033[4m{_train_size}\
\033[0m")
        _train, _test = get_score_for_spec_train_size(_train_size)
        _train_score_list.append(_train)
        _test_score_list.append(_test)
    rendu_graphique(
        _entier_list_train_size, _train_score_list, _test_score_list)


def rendu_graphique(
        _entier_list_train_size: list, _train_score_list: list,
        _test_score_list: list) -> None:
    """Creation des graphiques avec matplotlib.pyplot

    Args:
        _entier_list_train_size (list): taille des jeux de données
        _train_score_list (list): liste des scores d'entrainement
        _test_score_list (list): liste des scores de testes
    """
    plt.plot(
        _entier_list_train_size,
        _train_score_list,
        "-o",
        label='training score graphique')
    plt.plot(
        _entier_list_train_size,
        _test_score_list,
        "-o",
        label='test score graphique')
    plt.xlabel('train size')
    plt.ylabel('score')
    plt.grid(True)
    plt.show()


def comparaison_score_by_pathologie() -> None:
    """Test les scores de chaques pathologies
    """
    print("démarrage de la comparaison par score de pathologie")
    _cx_train, _cx_test, _pathologie_train, _pathologie_test =\
        creating_sets(None)
    _mlp = mlp_apprentissage(_cx_train, _pathologie_train)
    for _path_num in range(NUMBER_OF_PATH):
        _filtered_cx, _filtered_pathologie_test =\
            filter_cx_by_path(_cx_test, _pathologie_test, _path_num)
        _test_score = _mlp.score(_filtered_cx, _filtered_pathologie_test)
        print(
            f"score de test de la pathologie: {_path_num} -> \
{color_score(_test_score)}")


def color_score(_test_score: float) -> str:
    """Colorisation des valeurs de score dans la console

    Args:
        _test_score (float): score de test

    Returns:
        str: le score avec une couleur selon sa valeur
    """
    if _test_score <= 0.7:
        return f"\033[91m\033[1m{_test_score}\033[0m"
    return f"\033[92m\033[1m{_test_score}\033[0m"


def main() -> None:
    """Fonction de démarrage du module
    """
    print("part2")
    # Question 1 parti 2
    # comparaison_train_size()
    # Question 2 parti 2
    comparaison_score_by_pathologie()


if __name__ == "__main__":
    main()
