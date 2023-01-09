# TP2

## __Reponse Partie 1__ : Decouverte de la base


### __Q1.1__ : Que remarquez vous au sujet de nombre d’occurrences de chaque pathologie ?


 ![pathologie](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/répartition_pathologie.png)

_Figure: répartition des pathologies_

On remarque une répartition hétérogène des pathologies avec une très faible densité pour la pathologie 4.

> Cela risque de créer des problemes de classification pour la pathologie 4. 

### __Q1.2__ : De quelle forme est la distribution de l’âge ? Est-ce normal ?

 ![age](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/répartition_age.png)

 On a une gaussienne pour la répartition des âges.

## __Reponse Partie 2__ : Apprentissage sur la base 


### __Q2.1__ : Essayez de faire l’apprentissage avec différents volumes de données, à partir de quel volume il n’est plus utile de rajouter d’autres données ?

En faisaint de multiple apprentissage, en modifiant la taille du jeu d'entrainement on a :

 ![score_lbfgs](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/score_evolution_lbfgs.png)

_Figure: évolution de score en fonction de la taille d'entrainement, en bleu le score d'entrainement et en orange celui du test. Le solver ici est `lbfgs`._

 ![score_lbfgs](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/score_evolution_adam.png)

_Figure: évolution de score en fonction de la taille d'entrainement, en bleu le score d'entrainement et en orange celui du test. Le solver ici est `adam`._

Pour les deux solvers vers 2500 on a un score approchant le score de 0.95 et on commence à converger.

### __Q2.2__ : Dressez un score par pathologie, le score est-il le même pour chaque pathologie ? Pourquoi ?

Dans un premier temps l'étude des scores par pathologie va nous donner des scores très hétérogènes :

 ![score_lbfgs](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/score_comparaison_pathologie_lbfgs.png)

 _Figure : Captur de la console après deux essaies_

 On voit ici que les 4 premières pathologies (0 à 3) ont des résultats d'acceptables à bon, on est entre 0.77 et 0.99, alors que les 2 autres sont instables et aux alentours de 0 à 0.4.

 > La première conclusion tirée de ces tests c'est que certaines pathologies sont plus rares que d'autres. Cette rareté peut impliquer un manque d'entrainement et donc des résultats plutôt faibles.

 Cependant, dans un deuxième temps on va changer de solver. Sur les premièrs tests on a utilisé le solver `lbfgs`. On va faire les mêmes tests avec le solver `adam`.

 ![score_adam](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/score_comparaison_pathologie_adam.png)

> A noter que le solveur `lbfgs` a des erreurs d'itération max contrairement au solveur `adam`.
Par ailleurs le solveur adam va beaucoup plus vite sans erreur.

On remarque que les pathologies qui tout à l'heure étaient proches de 0 en score de tests sont maintenant aux alentours des 0.8 et 0.9.

> Il y a malgré tout une certaine instabilité sur la pathologie 4.

La suposition sur ce changement de score juste en changeant le solver, est que le jeu de donnée a une taille assez conséquente et le solveur `adam` est plus adapté à ce type contrairement au `lbfgs`.

## __Partie 3__ : Etude avancée

### __Q3.1__ : Les entrées sont-elles toutes utiles à apprentissage ?

Pour tester l'importance de chaque caractéritiques on va faire plusieurs entrainement en enlevant 1 caractéristique à chaque fois.

>On verra si le score est élevé, ça implique que la caractéristique n'est pas importante particulièrement et vis-versa
 - essai 1 :
 
 ![score_filtre_essaie_1](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/score_sans_une_caracteristique_1er_essai.png)
 
 - essai 2 :

 ![score_filtre_essaie_1](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/score_sans_une_caracteristique_1er_essai.png)

 On voit bien ici que la caractéristique la plus influente est la 6. D'ailleurs si on fait juste le test en filtrant toutes les autres caractéristiques on obtient environ 0.83 de score (entrainement et test)

 ![score_filtre_essaie_1](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/score_avec_juste_la_caractéristique_6.png)
 
 >C'est un score plutôt élevé malgré qu'on ait enlevé 7 caractéristiques.

### __Q3.2__ : À l’aide de la Q 2.2, essayez de déterminer si certaines entrées sont plus déterminantes sur l’apparition decertaines pathologies.

Le programme va ici filtrer chaque jeu de données avec juste une seule caractéristique et retourner le score de chaque pathlogie dans le cadre du test:

 ![score_path_par_carac1](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/score_path_par_carac1.png)

 ![score_path_par_carac2](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp2/ressources/score_path_par_carac2.png)

Les informations ressortant des résultats sont :

* L'age est déterminant pour toutes les pathologies sauf la 4 et moyennement pour la 3.
* La caractéristique 1 est déterminante pour toutes les pathologies sauf la 3 et la 4.
* La caractéristique 2 est déterminante pour toutes les pathologies sauf la 2 et moyennement la 4.
* La caractéristique 3 est déterminante pour toutes les pathologies surtout pour la 0, 1 et 2.
* La caractéristique 4 est déterminante pour toutes les pathologies sauf la 5 et moyennement la 4.
* La caractéristique 5 est déterminante pour toutes les pathologies.
* La caractéristique 6 est déterminante pour la pathologie 0 surtout et moyennement 3.
* La caractéristique 7 est déterminante pour toutes les pathologies sauf la 4.

Si on fait un bilan on voit que les caractéristiques 5 et 3 sont les caractéristiques les plus déterminantes pour toutes les pathologies.