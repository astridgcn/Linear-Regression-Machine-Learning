# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 21:28:53 2022

@author: As
"""
# Remarques :
# - Ce code n'est pas le plus optimisé, mais il est suffisamment simple pour illustrer le principe de l'algorithme.
# - On pourrait par exemple récupérer les lettres d'un document texte et les stocker dans une liste.
# - Il a été décidé (totalement arbitrairement) que les lettres bruitées ayant plus de 5 différences avec les autres lettres soient marquées comme non reconnues. (Sinon, elle est reconnnue comme celle avec qui elle a le moins grand nombre de différences).

# N'hésitez pas à proposer des améliorations et/ou extensions !

# ----------------- LIBRAIRIES -----------------

# 1. Librairies 

#Visualisation de données
from matplotlib import pyplot 
import matplotlib.pyplot as plt
#Tableaux
import numpy as np
#Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----------------- RÉGRESSION -----------------

# 2. Donnnées

## a) Création des données (vous pouvez charger des données existantes avec read_csv comme vu en cours)
age = list(range(1,19))
taille = [75, 85, 96, 100, 104, 110, 117, 120, 125, 130, 133, 140, 145, 150, 157, 160, 170, 175]
print("\n------------------ Données ------------------")
print("Âges : \n", age)
print("Tailles : \n",taille)

## b) Division des données en apprentissage et test

# On répartit les données aléatoirement : 80% en données d'apprentissage, et 20% pour tester (le text_size = 0.2)
age_train, age_test, taille_train, taille_test = train_test_split(age, taille, test_size=0.2, random_state=0)

# c) On les affiche
print("\n----------- Données d'entraînement ----------")
print("Âges : \n",age_train)
print("Tailles : \n",taille_train)
print("\n-------------- Données de test --------------")
print("Âges : \n",age_test)
print("Tailles : \n",taille_test)

# d) Transformation

# On transforme en vecteur colonne pour le modèle (ne change pas les données, seulement leur forme)
age_train = np.reshape(age_train, (-1, 1)) # Si on met (1, -1), il s'agira d'un vecteur colonne
#print(age_train) 

age_test = np.reshape(age_test, (-1, 1))
#print(age_test) 

# 3. Modèle

## a) Création du modèle

# On crée le modèle
model = LinearRegression()

# On remplit le modèle
model.fit(age_train, taille_train)

# On évalue ses performances
print("\n------------------ Modèle ------------------")
print("Performances : ", model.score(age_train, taille_train))

# On regarde les informations du modèle
print("Coefficient(s) du modèle :",  model.coef_)
print("Ordonnée à l'origine du modèle :", model.intercept_)

## b) Prédiction

# On teste la capacité du modèle à prédire (les fameux 20% de données qu'on a mises de côté pour tester notre modèle)
print("\nPrédictions du modèle :", "\nTailles qui doivent être prédites :", taille_test, "\nTailles prédites par le modèle :", model.predict(age_test))
