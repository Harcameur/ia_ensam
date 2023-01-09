# TP1

## Reponse Partie 2

---

### Q2.1 Changez le nombre de neurones de la couche cachée à 20, puis 10, puis 5, puis 1, qu'observez vous ?

---
En changeant le nombre de neuronnes on obtient:
- 50 :
 ![50](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/multiple%20neurone/50.png)
- 20 :
 ![20](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/multiple%20neurone/20.png)
- 10 :
 ![10](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/multiple%20neurone/10.png)
- 5 :
 ![5](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/multiple%20neurone/5.png)
- 1 :
 ![1](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/multiple%20neurone/1.png)

On observe que les lignes délimitants la zone verte deviennent de plus en plus cassantes, on a une zone qui épouse moins la forme des points.

---

### Q2.2 Avec 50 neurones essayez les 4 fonctions d’activation, laquelle ne convient pas à ces données ? Pourquoi ?

---

En changeant les fonctions d'activation on obtient :

- `identity` ![Identity image](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/fonction%20activations/identity.png)

- `logistic` ![logistic image](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/fonction%20activations/logistic.png)

- `relu` ![Relu image](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/fonction%20activations/relu.png)

- `tanh` ![Tanh image](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/fonction%20activations/tanh.png)

La fonction d'activation `identity` semble avoir des soucis à afficher une délimitation correcte.

---

### Q2.3 Comment pourrait-on résoudre cette problématique avec un seul neurone dans la couche cachée ?

---


En testant sur tenserflow playground un cas à peu près similaire, en mettant les valeurs d'entrées au carré on obtient:


![tsf image](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/tsf.png)


Ainsi, si on le met au carré sur notre programme on a :

![Q23 image](https://raw.githubusercontent.com/Harcameur/ia_ensam/main/src/tp1/ressources/Q23.png)