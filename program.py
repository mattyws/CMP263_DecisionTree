import pandas as pd
import random
import math

data = pd.read_csv("resources/dadosBenchmark_validacaoAlgoritmoAD.csv", sep=';')
# vejo os cinco primeiros elementos dos meus dados
print(data.head())
# vejo todos os meus elementos da coluna Tempo
print(data["Tempo"])
# me mostra quais os tipos dos meus dados
print(data.dtypes)
# seleciona a primeira linha dos meus dados
print(data.iloc[[2]])
# retorna a quantidade de linhas da minha matriz
print(len(data))

# semente para o número aleatório
random.seed(1)
# retorna um valor dentre 0 e o tamanho dos meus dados
print(random.randrange(len(data)))

# tamanho do meu DataFrame
tam = len(data)

# crio novo DataFrame
new_df = pd.DataFrame()

# cria bootstrap como um DataFrame de mesmo tamanho dos dados originais
for i in range(tam):
    new_df = new_df.append(data.iloc[[random.randrange(tam)]], ignore_index=True)

print(new_df)

# mostra o número de colunas que há no meu DataFrame
# -1 pois a coluna do atributo alvo não conta
total_atr = len(data.columns)-1
print(total_atr)

# pego a raiz quadrada do total de colunas e arredondo para baixo (ou deveria ser para cima?)
m = math.floor(math.sqrt(total_atr))
print(m)


most_new_df = pd.DataFrame()

# tenho que tirar a coluna alvo do DataFrame
new_df = new_df[new_df.columns[0: len(new_df.columns)-1]]
print(new_df)

print("..........\n")
# seleciona quais com quais atributos irei trabalhar
for i in range(m):
    col = random.randrange(len(new_df.columns))+1
    most_new_df = pd.concat([most_new_df, new_df[new_df.columns[col-1:col]] ], axis = 1)


print(most_new_df)
print(most_new_df[:3])