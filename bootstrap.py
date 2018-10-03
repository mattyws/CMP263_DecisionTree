import pandas as pd
import random
import math

class Bootstrap:

    @staticmethod
    def make_bootstrap(data, labels):
        """
        Cria o bootstrap a partir dos dados originais
        :param data: dados originais contendo a coluna de labels
        :return: dados e labels gerados a partir do bootstrap
        """

        # tamanho do meu DataFrame
        tam = len(data)

        # crio novo DataFrame
        new_df = pd.DataFrame()
        new_labels = pd.DataFrame()

        # cria bootstrap como um DataFrame de mesmo tamanho dos dados originais
        for i in range(tam):
            rand_num = random.randrange(tam)
            new_df = new_df.append(data.iloc[rand_num], ignore_index=True)
            new_labels = new_labels.append(labels.iloc[rand_num], ignore_index=True)

        return new_df, new_labels


    @staticmethod
    def select_columns(data, m = -1):
        """
        Seleciona quais as colunas do bootstrap devem ser verificadas pela floresta
        :param data: a base de dados contendo a coluna sem labels
        :param m: quantas colunas serão pegas para selecionar, caso não seja inserido será utilizado a raiz quadrada
        """
        # mostra o número de colunas que há no meu DataFrame
        total_atr = len(data.columns)

        # caso o usuário não coloque nenhum valor para m, então ele irá selecionar a raiz quadrada do total de colunas
        if (m == -1):
            # pego a raiz quadrada do total de colunas e arredondo para baixo (ou deveria ser para cima?)
            m = math.floor(math.sqrt(total_atr))

        new_df = pd.DataFrame()

        # seleciona quais com quais atributos irei trabalhar
        for i in range(m):
            col = random.randrange(len(data.columns)) + 1
            new_df = pd.concat([new_df, data[data.columns[col - 1:col]]], axis=1)
            # remover a coluna que foi selecionada para estar em new_df
            data = data.drop(data[data.columns[col - 1:col]], axis = 1)

        return new_df