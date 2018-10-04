import csv
import math
import operator

import pandas as pd
import numpy as np

class DecisionNode(object):
    """
    The nodes from a decision tree.
    Each node is composed by the column which the decision is done, and their child.
    """
    def __init__(self, column, is_terminal=False, is_numeric=False, parent=None):
        """
        Constructor for the DecisionNode.
        :param column: the column what the node will use to make its decision. If is a terminal, the value is the class that will be returned
        :param is_terminal: if the node is a terminal node
        """
        self.column = column
        self.is_terminal = is_terminal
        # If it's terminal, it doesn't have a child
        self.child = dict()
        self.is_numeric = is_numeric
        self.parent = parent

    def add_child(self, value, node):
        """
        Add a child node to the current node.
        :param value: the value from that leads to the child node
        :param node: the child node
        """
        if not self.is_terminal:
            self.child[value] = node

    def get_child_node(self, value):
        """
        Get the child node from a specific value. Throws a value error if the value do not exists in the node.
        :param value: the value to get the child from
        :return: the child node
        """
        if not self.is_terminal:
            if self.is_numeric:
                keys = list(self.child.keys())
                median = float(keys[0].split()[1])
                if value < median:
                    sign = "<"
                else:
                    sign = ">"
                for key in keys:
                    if sign in key:
                        return self.child[key]
            else:
                if value in self.child.keys():
                    return self.child[value]
                else:
                    raise ValueError("Value {0} not in the tree!".format(value))

    def get_value(self):
        """
        If the node is a terminal, return the class that will be predicted
        :return: the value
        """
        if self.is_terminal:
            return self.column
        else:
            raise ValueError("Non-terminal nodes doesn't have a value.")

    def __len__(self):
        if self.is_terminal:
            return 0
        return len(self.child.keys())

    def __str__(self):
        return self.column



class DecisionTree(object):
    """
    A class for building a Decision Tree.
    All data passed to this class has to be a pandas DataFrame.
    """

    def __init__(self):
        """
        The constructor.
        """
        self.root = None

    def train(self, X, Y, max_depth=3):
        """
        The training algorithm for the decision tree.
        :param X: the data to build the tree.
        :param Y: the labels for the data.
        :param max_depth: the max depth to build the tree
        """
        if len(X) == 0:
            # Data can't be empty
            raise ValueError("Data cannot be empty.")
        if len(Y) != len(X):
            # Labels have to have the same size than data
            raise ValueError("Data and the labels has to have the same size.")
        self.max_depth = max_depth
        # Start building the tree
        self.root = self.__build(X, Y)

    def predict(self, data):
        """
        Predict labels for a given set of data.
        :param data: the data to predict the labels for.
        :return: a class if only one row is given, a array of classes if a set of rows is given.
        """
        if data is not None and len(data) != 0:
            if data.shape == (1, len(data.columns)):
                # If the data has only one row, just return the output of the tree
                node = self.root
                while not node.is_terminal:
                    # Do the walk on the tree until the node is a terminal node
                    node = node.get_child_node(data[node.column].item())
                #Return the value of the terminal node
                return node.get_value()
            else:
                # If has more than one row, do the predicting for all data given
                classes = []
                for index, row in data.iterrows():
                    node = self.root
                    while not node.is_terminal:
                        print(node.column)
                        node = node.get_child_node(row[node.column])
                    classes.append(node.get_value())
                # Return all classes predicted for each data
                return classes


    def __build(self, data, labels, depth=0, parent=None):
        """
        The algorithm that build the decision tree.
        It is a recursive algorithm that builds it top-down.
        :param data: the data to build the node
        :param labels: the labels for the data
        :param depth: the current depth for the tree
        :return: the node for the given snapshot of the data
        """
        # Get the information gain for each column on data
        info_gain_columns = self.__info_gain(data, labels)
        # Get the higher information gain obtained
        best_info_column = max(info_gain_columns, key=info_gain_columns.get)
        # print("============================== {0} ===============================".format(best_info_column))
        # Get the uniques values existent, if it's numeric, get the unique values for the cut
        best_column_is_numeric = self.__is_numeric(data, best_info_column)
        if best_column_is_numeric:
            values = self.__numeric_column(data[best_info_column]).unique()
            node = DecisionNode(best_info_column, is_numeric=True, parent=parent)
        else:
            values = data[best_info_column].unique()
            node = DecisionNode(best_info_column, parent=parent)
        # Create a node for the best information gain column
        # Now we have the current depth where the new node created is
        depth += 1
        for value in values:
            """
            Looping each value over the unique values existent on the best column.
            For each value, filter a new snapshot using the given value, and call __build using the snapshot to
            create the node for that snapshot.
            """
            # Get the new snapshot for that value
            new_data, new_labels = self.__get_split(data, labels, best_info_column, value, is_numeric=best_column_is_numeric)
            # Calculate the entropy for the new snapshot
            value_entropy = self.__entropy(new_labels[new_labels.columns[0]].get_values())
            if (self.max_depth > depth and value_entropy != 0) or (self.max_depth <= depth and value_entropy == 0.5):
                """
                If we do not reach the max depth and the entropy is not 0, or if we reach the max depth but the
                classes are even distributed (entropy == 0.5) call build and create a new node for the value
                """
                new_node = self.__build(new_data, new_labels, depth=depth, parent=best_info_column)
                # Add the new created child to the current node
                node.add_child(value, new_node)
            elif value_entropy == 0 or (self.max_depth <= depth and value_entropy != 0):
                # If the value of the entropy is 0, the snapshot is pure, create a terminal node
                terminal_node = DecisionNode(self.__get_higher_frequency_value(new_labels), is_terminal=True, parent=best_info_column)
                # Add  the terminal node as a child of the current node
                node.add_child(value, terminal_node)
        return node

    def __info_gain(self, data, labels):
        """
        Compute the information gain for each column
        :param data: the data
        :param labels: the class for each row in data
        :return: a dictionary with "column": InformationGain
        """
        # Get the entropy for the labels
        infoD = self.__entropy(labels[labels.columns[0]].get_values())
        info_columns = dict()
        for j in data.columns:
            data_column = data[j]
            if self.__is_numeric(data, j):
                data_column = self.__numeric_column(data_column)
            # Compute the entropy for each value
            column_values = dict()
            for i in range(len(data_column)):
                # Compute the appearance for each value in the column
                if data_column[i] not in column_values.keys():
                    column_values[data_column[i]] = []
                column_values[data_column[i]].append(i)
            entropy_column = 0
            for key in column_values.keys():
                # Compute the entropy for each value in the column
                slice_labels = []
                for value in column_values[key]:
                    slice_labels.append(labels[labels.columns[0]][value])
                infoDj = self.__entropy(slice_labels)
                entropy_column += ( len(column_values[key])/ len(data) ) * infoDj
            info_columns[j] = entropy_column
        for i in info_columns.keys():
            # Now compute the information gain for each column
             info_columns[i] = infoD - info_columns[i]
        return info_columns


    def __entropy(self, column):
        """
        Compute the entropy for a given column
        :param column:
        :return: the entropy
        """
        label_values = dict()
        for value in column:
            if value not in label_values.keys():
                label_values[value] = 0
            label_values[value] += 1
        entropy = 0
        for key in label_values:
            entropy += -((label_values[key] / len(column)) * math.log2(label_values[key] / len(column)))
        return entropy

    def __numeric_column(self, column):
        new_column = sorted(column)
        if len(new_column)%2 == 0:
            median = new_column[len(new_column)//2]
        else:
            median = (new_column[len(new_column)//2] + new_column[(len(new_column)//2)+1])/2
        new_column = pd.cut(column, bins=[0, median, np.inf], labels=["<= {}".format(median), "> {}".format(median)], right=True)
        return new_column

    def __get_split(self, data, labels, row, value, is_numeric=False):
        """
        Get a split for the data given the row and the value.
        :param data: the data to use for split
        :param labels: the labels for that data
        :param row: the row name to get the value from
        :param value: the value used to filter
        :return: the splited data, labels
        """
        # Filter all data in the row that are equal to the value
        if is_numeric:
            median = float(value.split()[1])
            if ">" in value:
                new_data = data[data[row] > median]
            else:
                new_data = data[data[row] <= median]
        else:
            new_data = data[data[row] == value]
        # Get the labels for those filtered data, using it index
        new_labels = labels.iloc[new_data.index.tolist()]
        # Reset the indexes for both, that is done because pandas uses the index got from the original data
        new_data = new_data.reset_index(drop=True)
        new_labels = new_labels.reset_index(drop=True)
        return new_data, new_labels

    def __get_higher_frequency_value(self, labels):
        """
        Get the value with higher frequency
        :param labels: the labels to compute the frequency
        :return: the label with higher frequency
        """
        return labels[labels.columns[0]].value_counts().idxmax()

    def __is_numeric(self, data, column):
        return data[column].dtype == "int64" or data[column].dtype == "float64"

    def print(self, graphviz=False, filename="tree"):
        """
        Print the tree in a pretty way.
        """
        if graphviz :
            self.__render_graphviz_tree(filename=filename)
        else:
            self.__print_tree(self.root)

    def __print_tree(self, current_node, indent="", last='updown'):
        """
        Print the tree.
        The code was gotten from https://stackoverflow.com/questions/30893895/how-to-print-a-tree-in-python
        :param current_node: the current node
        :param indent: how the ident was done
        :param last: where the last node is
        """
        nb_children = lambda node: len(node) + 1
        size_branch = {child: nb_children(current_node.child[child]) for child in current_node.child.keys()}

        """ Creation of balanced lists for "up" branch and "down" branch. """
        up = sorted(current_node.child, key=lambda node: nb_children(node))
        down = []
        while up and sum(size_branch[node] for node in down) < sum(size_branch[node] for node in up):
            down.append(up.pop())

        """ Printing of "up" branch. """
        for child in up:
            next_last = 'up' if up.index(child) is 0 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else '│', " " * len(current_node.column))
            self.__print_tree(current_node.child[child], indent=next_indent, last=next_last)

        """ Printing of current node. """
        if last == 'up':
            start_shape = '┌'
        elif last == 'down':
            start_shape = '└'
        elif last == 'updown':
            start_shape = ' '
        else:
            start_shape = '├'

        if up:
            end_shape = '┤'
        elif down:
            end_shape = '┐'
        else:
            end_shape = ''

        print('{0}{1}{2}{3}'.format(indent, start_shape, current_node.column, end_shape))

        """ Printing of "down" branch. """
        for child in down:
            next_last = 'down' if down.index(child) is len(down) - 1 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'down' in last else '│', " " * len(current_node.column))
            self.__print_tree(current_node.child[child], indent=next_indent, last=next_last)

    def __render_graphviz_tree(self, filename="tree"):
        import graphviz
        nodes = [self.root]
        tree = graphviz.Digraph(format="png", filename=filename)
        tree.attr("node", shape="box")
        id = 0
        nodes = [ {"node":self.root, "parent_id":None, "id":id, "key":None} ]
        for node in nodes:
            for key in node["node"].child.keys():
                id+=1
                nodes.append( {"node":node["node"].child[key], "parent_id":node["id"], "id":id, "key":key} )
        for node in nodes:
            tree.node(str(node["id"]), label=str(node["node"].column))
            if node["parent_id"] is not None:
                tree.edge(str(node["parent_id"]), str(node["id"]), label=node["key"])
        tree.render()

if __name__ == "__main__":
    data = []
    labels = []
    data = pd.read_csv('/home/mattyws/Documentos/DecisionTrees/CMP263_DecisionTree/resources/dadosBenchmark_validacaoAlgoritmoAD.csv', sep=';')
    labels = data.drop(columns=data.columns[:-1])
    data = data.drop(columns=data.columns[-1])
    tree = DecisionTree()
    tree.train(data, labels)
    tree.print(graphviz=True, filename="benchmark_tree/tree")
