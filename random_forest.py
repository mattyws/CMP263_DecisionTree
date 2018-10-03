import bootstrap
import decision_tree
from statistics import mode


class RandomForest(object):

    def __init__(self):
        self.trees = []

    def train(self, X, Y, num_trees=3, max_depth=3):
        for i in range(num_trees):
            tree_bootstrap, labels = bootstrap.Bootstrap.make_bootstrap(X, Y)
            tree_bootstrap = bootstrap.Bootstrap.select_columns(tree_bootstrap)
            tree = decision_tree.DecisionTree()
            tree.train(tree_bootstrap, labels)
            self.trees.append(tree)
            # tree.print(graphviz=True, filename="tree_{}".format(i))

    def predict(self, data):
        if data is not None and len(data) != 0:
            if data.shape == (1, len(data.columns)):
                # If the data has only one row, just return the consensus from the trees
                predicted_classes = []
                for tree in self.trees:
                    predicted_classes.append(tree.predict(data))
                return mode(predicted_classes)
            else:
                # If has more than one row, do the predicting for all data given
                classes = []
                for index, row in data.iterrows():
                    predicted_classes = []
                    for tree in self.trees:
                        predicted_classes.append(tree.predict(data))
                    classes.append(mode(predicted_classes))
                return classes

    def get_trees_prediction(self, data):
        if data is not None and len(data) != 0:
            if data.shape == (1, len(data.columns)):
                # If the data has only one row, just return the consensus from the trees
                predicted_classes = []
                for tree in self.trees:
                    predicted_classes.append(tree.predict(data))
                return predicted_classes
            else:
                # If has more than one row, do the predicting for all data given
                classes = []
                for index, row in data.iterrows():
                    predicted_classes = []
                    for tree in self.trees:
                        predicted_classes.append(tree.predict(data))
                    classes.append(predicted_classes)
                return classes

    def print_forest_trees(self, sufix="tree", graphviz=False):
        i = 0
        for tree in self.trees:
            tree.print(graphviz=graphviz, filename=sufix+"_{}".format(i))
            i+=1

