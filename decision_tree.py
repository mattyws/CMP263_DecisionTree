import csv
import math
import operator

import pandas as pd

class DecisionNode(object):
    def __init__(self, column, values):
        self.column = column
        self.values = values
        self.child = []



class DecisionTree(object):

    def __init__(self):
        self.data = None
        self.labels = None

    def train(self, X, Y, max_depth=3):
        if len(X) == 0:
            raise ValueError("Data cannot be empty.")
        if len(Y) != len(X):
            raise ValueError("Data and the labels has to have the same size.")
        self.data = X
        self.labels = Y
        self.max_depth = max_depth
        self.__build(self.data, self.labels)

    def __build(self, data, labels):
        info_gain_columns = self.__info_gain(data, labels)
        print(info_gain_columns)
        best_info_column = max(info_gain_columns, key=info_gain_columns.get)
        values = data[best_info_column].unique()
        for value in values:
            new_data, new_labels = self.__get_split(data, labels, best_info_column, value)
            print(new_data)
            print(new_labels)
            self.__build(new_data, new_labels)
        pass

    def __info_gain(self, data, labels):
        """
        Compute the information gain for each column
        :param data: the data
        :param labels: the class for each row in data
        :return: a dictionary with "column": InformationGain
        """
        infoD = self.__infoD(self.labels[labels.columns[0]].get_values())
        info_columns = dict()
        for j in data.columns:
            column_values = dict()
            for i in range(len(data[j])):
                if data[j][i] not in column_values.keys():
                    column_values[data[j][i]] = []
                column_values[data[j][i]].append(i)
            info_column = 0
            for key in column_values.keys():
                slice_labels = []
                for value in column_values[key]:
                    slice_labels.append(labels[labels.columns[0]][value])
                infoDj = self.__infoD(slice_labels)
                info_column += ( len(column_values[key])/ len(data) ) * infoDj

            info_columns[j] = info_column
        for i in info_columns.keys():
             info_columns[i] = infoD - info_columns[i]
        return info_columns

    def __infoD(self, labels):
        label_values = dict()
        for value in labels:
            if value not in label_values.keys():
                label_values[value] = 0
            label_values[value] += 1
        infoD = 0
        for key in label_values:
            infoD += -( (label_values[key]/len(labels))*math.log2(label_values[key]/len(labels)) )
        return infoD

    def __get_split(self, data, labels, row, value):
        new_data = data[data[row] == value]
        new_labels = labels.iloc[new_data.index.tolist()]
        new_data = new_data.reset_index(drop=True)
        new_labels = new_labels.reset_index(drop=True)
        return new_data, new_labels


if __name__ == "__main__":
    data = []
    labels = []
    # with open('/home/mattyws/Documentos/DecisionTrees/CMP263_DecisionTree/resources/dadosBenchmark_validacaoAlgoritmoAD.csv', 'r') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=';')
    #     for row in spamreader:
    #         data.append(row[:-1])
    #         labels.append(row[-1])
    data = pd.read_csv('/home/mattyws/Documentos/DecisionTrees/CMP263_DecisionTree/resources/dadosBenchmark_validacaoAlgoritmoAD.csv', sep=';')
    labels = data.drop(columns=data.columns[:-1])
    data = data.drop(columns=data.columns[-1])
    tree = DecisionTree()
    tree.train(data, labels)
