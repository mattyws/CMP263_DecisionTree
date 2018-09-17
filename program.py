import pandas as pd
import random

import bootstrap

data = pd.read_csv("dadosBenchmark_validacaoAlgoritmoAD.csv", sep=';')
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

# semente para o número aleatório
random.seed(100)
# retorna um valor dentre 0 e o tamanho dos meus dados
#print(random.randrange(len(data)))

new_df = bootstrap.Bootstrap.make_bootstrap(data)
print(new_df)

new_df = bootstrap.Bootstrap.select_columns(new_df)
print(new_df)