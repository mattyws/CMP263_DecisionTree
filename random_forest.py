import bootstrap
import decision_tree


class RandomForest(object):

    def __init__(self):
        self.trees = []

    def train(self, X, Y, num_trees=3, max_depth=3):
        for i in range(num_trees):
            tree_bootstrap = bootstrap.Bootstrap.make_bootstrap(X)
            tree = decision_tree.DecisionTree()
            tree.train(tree_bootstrap, )