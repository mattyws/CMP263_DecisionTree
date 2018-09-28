import pandas as pd
import random

import bootstrap
import decision_tree
import KFold

data_wine = []
labels_wine = []

#data_wine parecem ser todos dados numéricos
data_wine = pd.read_csv("/home/hortensia/Documentos/TrabalhoML1/CMP263_DecisionTree/resources/Wine/wine.data.txt")

# Cria colunas
data_wine.columns = ["TipoVinho","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

# semente para o número aleatório
random.seed(100)

k = 10

kf = KFold.KFold()
kf.make_kfold(data_wine, 0, k, 100)

data_test, data_train = kf.get_data_test_train()
#print(data_train)
#print("Tamanho dados originais: ", len(data_wine))
#print("Tamanho dados de treinamento: ",len(data_train))
#print("Tamanho dados de teste: ", len(data_test))


# cria bootstrap a partir dos dados de treinamento
new_df_wine = bootstrap.Bootstrap.make_bootstrap(data_train)
#print(new_df_wine)

#retirar o label do data_wine depois de ter gerado o bootstrap
labels_wine = new_df_wine[new_df_wine.columns[0:1]]
new_df_wine = new_df_wine.drop(new_df_wine[new_df_wine.columns[0:1]], axis = 1)

new_df_wine = bootstrap.Bootstrap.select_columns(new_df_wine)
#print("Imprimir dados com colunadas selecionadas")
#print(new_df_wine)

tree_wine = decision_tree.DecisionTree()
tree_wine.train(new_df_wine, labels_wine)

#print(data_wine.iloc[0], labels_wine.iloc[0])
#print(tree_wine.predict(data_wine.iloc[[0]]))
#tree_wine.print()
tree_wine.print(graphviz=True)