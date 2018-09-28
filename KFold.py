import pandas as pd
import numpy as np



class KFold:

    data_kfold = []

    indice_teste = 0


    def make_kfold(self, data, col_labels, k, seed):
        """
        Cria k folds
        :param data: a base de dados contendo a coluna com labels
        :param col_labels: indica qual a coluna que possui os labels
        :param k: quantidade de folds
        :param seed: semente para aleatorização
        """

        np.random.seed(seed)

        data = data.iloc[np.random.permutation(len(data))]

        labels = data.iloc[:, col_labels]

        grouped = data.groupby(labels)

        new_df = pd.DataFrame()
        for key, item in grouped:
            g = grouped.get_group(key)
            new_df = new_df.append(g, ignore_index=True)

        df_array = []
        for j in range(k):
            df_array.append(pd.DataFrame())

        j = 0
        for i in range(len(new_df)):
            if j >= k:
                j = 0
            df_array[j] = df_array[j].append(new_df.iloc[[i]], ignore_index=True)
            j += 1

        self.data_kfold = df_array
        self.indice_teste = 0


    def get_data_test_train(self):
        """
        Retorna para o usuário quais são os dados de testes e de treinamento atuais
        :return::return um DataFrame de teste e um DataFrame com os dados de treinamento
        """

        data_test = self.data_kfold[self.indice_teste]
        data_train = pd.DataFrame()

        for j in range(len(self.data_kfold)):
            if j != self.indice_teste:
                data_train = pd.concat([data_train, self.data_kfold[j]])
                #new_df = pd.concat([new_df, data[data.columns[col - 1:col]]], axis=1)
                #data_train[j] = data_train[j].append(self.data_kfold[j])

        return (data_test, data_train)


    def update_indice_teste(self):
        self.indice_teste += 1
