import csv
import math
import operator

import pandas as pd
#TODO 1: Comment
class DecisionNode(object):
    def __init__(self, column, is_terminal=False):
        self.column = column
        self.is_terminal = is_terminal
        if is_terminal:
            self.child = None
        else:
            self.child = dict()

    def add_child(self, value, node):
        if not self.is_terminal:
            self.child[value] = node

    def get_child_node(self, value):
        if not self.is_terminal:
            if value in self.child.keys():
                return self.child[value]
            else:
                raise ValueError("Value {0} not in the tree!".format(value))

    def get_value(self):
        if self.is_terminal:
            return self.column
        else:
            raise ValueError("Non-terminal nodes doesn't have a value.")



class DecisionTree(object):

    def __init__(self):
        self.data = None
        self.labels = None
        self.root = None

    def train(self, X, Y, max_depth=3):
        if len(X) == 0:
            raise ValueError("Data cannot be empty.")
        if len(Y) != len(X):
            raise ValueError("Data and the labels has to have the same size.")
        self.data = X
        self.labels = Y
        self.max_depth = max_depth
        self.root = self.__build(self.data, self.labels)
        print(type(self.root))

    def predict(self, data):
        print(data.shape)
        if data.shape == (1, len(data.columns)):
            node = self.root
            while not node.is_terminal:
                node = node.get_child_node(data[node.column][0])
            return node.get_value()
        else:
            classes = []
            for index, row in data.iterrows():
                node = self.root
                while not node.is_terminal:
                    node = node.get_child_node(row[node.column])
                classes.append(node.get_value())
            return classes


    def __build(self, data, labels, depth=0):
        info_gain_columns = self.__info_gain(data, labels)
        print(info_gain_columns)
        best_info_column = max(info_gain_columns, key=info_gain_columns.get)
        print("============================== {0} ===============================".format(best_info_column))
        values = data[best_info_column].unique()
        node = DecisionNode(best_info_column)
        depth += 1
        for value in values:
            print("xxxxxxxxxxxxxxxxxxxxxx {0} xxxxxxxxxxxxxxxxxxx".format(value))
            new_data, new_labels = self.__get_split(data, labels, best_info_column, value)
            print(new_data)
            print(new_labels)
            print("Calculating entropy for {0}".format(value))
            value_entropy = self.__entropy(new_labels[new_labels.columns[0]].get_values())
            print("------------------------ {0} entropy is 0".format(value))
            print(self.__get_higher_frequency_value(new_labels))
            if self.max_depth > depth and value_entropy != 0:
                new_node = self.__build(new_data, new_labels, depth=depth)
                node.add_child(value, new_node)
            elif value_entropy == 0:
                terminal_node = DecisionNode(self.__get_higher_frequency_value(new_labels), is_terminal=True)
                node.add_child(value, terminal_node)
            #TODO 2: max_depth == depth and value_entropy == 0.5
        return node

    def __info_gain(self, data, labels):
        """
        Compute the information gain for each column
        :param data: the data
        :param labels: the class for each row in data
        :return: a dictionary with "column": InformationGain
        """
        infoD = self.__entropy(labels[labels.columns[0]].get_values())
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
                infoDj = self.__entropy(slice_labels)
                info_column += ( len(column_values[key])/ len(data) ) * infoDj

            info_columns[j] = info_column
        for i in info_columns.keys():
             info_columns[i] = infoD - info_columns[i]
        return info_columns

    def __entropy(self, labels):
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

    def __get_higher_frequency_value(self, labels):
        return labels[labels.columns[0]].value_counts().idxmax()



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
    print(data.iloc[0], labels.iloc[0])
    print(tree.predict(data.iloc[[0]]))
