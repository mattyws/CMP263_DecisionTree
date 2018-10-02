import pandas as pd
import random

import bootstrap
import decision_tree
import kfold

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
#print(data_train.head(2))
#print("labels_train")
#print(labels_train.head(3))
#print("Tamanho dados originais: ", len(data_wine))
#print("Tamanho dados de treinamento: ", len(data_train))
#print("Tamanho dados de teste: ", len(data_test))


# cria bootstrap a partir dos dados de treinamento
new_df_wine, labels_wine = bootstrap.Bootstrap.make_bootstrap(data_train, labels_train)
#print(new_df_wine)

bootstrap_train = bootstrap.Bootstrap.select_columns(data_train)
print("Imprimir dados com colunadas selecionadas")
print(bootstrap_train)

#tree_wine = decision_tree.DecisionTree()
#tree_wine.train(bootstrap_train, labels_train)

#print(data_wine.iloc[0], labels_wine.iloc[0])
#print(tree_wine.predict(data_wine.iloc[[0]]))
#tree_wine.print()
#tree_wine.print(graphviz=True)