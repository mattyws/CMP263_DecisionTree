import pandas as pd
import random

import bootstrap
import decision_tree

#data = pd.read_csv("dadosBenchmark_validacaoAlgoritmoAD.csv", sep=';')
# vejo os cinco primeiros elementos dos meus dados
#print(data.head())
# vejo todos os meus elementos da coluna Tempo
#print(data["Tempo"])
# me mostra quais os tipos dos meus dados
#print(data.dtypes)
# seleciona a primeira linha dos meus dados
#print(data.iloc[[2]])
# retorna a quantidade de linhas da minha matriz
#print(len(data))



data_wine = []
labels_wine = []

#data_wine parecem ser todos dados numéricos
data_wine = pd.read_csv("/home/hortensia/Documentos/TrabalhoML1/CMP263_DecisionTree/resources/Wine/wine.data.txt")
#retirar o label do data_wine
labels_wine = data_wine[data_wine.columns[0:1]]
data_wine = data_wine.drop(data_wine[data_wine.columns[0:1]], axis = 1)

data_wine.columns = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

#print("imprimindo dados:")
#print(data_wine.head())
#print("imprimindo labels:")
#print(labels_wine.head())

#tree_wine = decision_tree.DecisionTree()
#tree_wine.train(data_wine, labels_wine)

#print(data_wine.iloc[0], labels_wine.iloc[0])
#print(tree_wine.predict(data_wine.iloc[[0]]))
#tree_wine.print()


# semente para o número aleatório
random.seed(100)
# retorna um valor dentre 0 e o tamanho dos meus dados
#print(random.randrange(len(data)))


new_df = bootstrap.Bootstrap.make_bootstrap(data_wine)
print(new_df)

new_df = bootstrap.Bootstrap.select_columns(new_df)
print(new_df)