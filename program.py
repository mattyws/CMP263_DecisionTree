import pandas as pd
import random

import bootstrap
import decision_tree
import kfold
from random_forest import RandomForest

data_wine = []
labels_wine = []

#data_wine parecem ser todos dados numéricos
data_wine = pd.read_csv("resources/Wine/wine.data.txt")

# Cria colunas
data_wine.columns = ["TipoVinho","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]


labels_wine = data_wine[data_wine.columns[0:1]]
data_wine = data_wine.drop(data_wine[data_wine.columns[0:1]], axis = 1)

# semente para o número aleatório
nseed = 100
random.seed(nseed)

k = 10

kf = kfold.KFold()
kf.make_kfold(data_wine, labels_wine, k, nseed)

data_test, labels_test, data_train, labels_train = kf.get_data_test_train()

forest = RandomForest()
forest.train(data_train, labels_train)