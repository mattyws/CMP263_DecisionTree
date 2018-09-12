import csv
import math

class Node(object):

    def __init__(self, values):
        self.values = values



class DecisionTree(object):

    def __init__(self):
        self.has_head = False
        self.data = None
        self.head = None
        self.labels = None
        self.columns_description = None

    def train(self, X, Y, columns_description, has_head=True):
        if len(X) == 0:
            raise ValueError("Data cannot be empty.")
        if len(Y) != len(X):
            raise ValueError("Data and the labels has to have the same size.")
        if len(X[0]) != len(columns_description):
            raise ValueError("The column descriptions has to have the same size of columns in data.")
        if has_head:
            self.data = X[1:]
            self.head = X[0]
            self.labels = Y[1:]
            self.head.append(Y[0])
        else:
            self.data = X
            self.labels = Y
        self.columns_description = columns_description
        self.has_head = has_head
        self.__build()

    def __build(self):
        info_gain_columns = self.__info_gain(self.data, self.labels)
        print(info_gain_columns)
        pass

    def __info_gain(self, data, labels):
        infoD = self.__infoD(self.labels)
        info_columns = []
        for j in range(len(data[0])):
            column_values = dict()
            for i in range(len(data)):
                if data[i][j] not in column_values.keys():
                    column_values[data[i][j]] = []
                column_values[data[i][j]].append(i)
            info_column = 0
            for key in column_values.keys():
                slice_labels = []
                for value in column_values[key]:
                    slice_labels.append(labels[value])
                infoDj = self.__infoD(slice_labels)
                info_column += ( len(column_values[key])/ len(data) ) * infoDj
            info_columns.append(info_column)
        for i in range(len(info_columns)):
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


if __name__ == "__main__":
    data = []
    labels = []
    with open('/home/mattyws/Documentos/DecisionTrees/CMP263_DecisionTree/resources/dadosBenchmark_validacaoAlgoritmoAD.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            data.append(row[:-1])
            labels.append(row[-1])
        tree = DecisionTree()
        tree.train(data, labels, ["categoric", "categoric", "categoric", "categoric"])
