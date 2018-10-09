import csv
import os
import pickle
from time import time

import pandas as pd
import random

import bootstrap
import decision_tree
import kfold
from random_forest import RandomForest
import collections
from helper import *


#WINE
# data = []
# labels = []
# data = pd.read_csv("resources/Wine/wine.data.txt", header=None, sep=',')
# # Cria colunas
# data.columns = ["TipoVinho", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
#                      "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
#                      "OD280/OD315 of diluted wines", "Proline"]
#
#
# labels = data[data.columns[0:1]]
# data = data.drop(data[data.columns[0:1]], axis=1)

# #IONOSPHERE
data = pd.read_csv("resources/Ionosphere/ionosphere.data.txt", header=None, sep=',', prefix='H')
labels = data.drop(columns=data.columns[:-1])
data = data.drop(columns=data.columns[-1])

#BrestCancer
# data = pd.read_csv("resources/BreastCancer/breast-cancer-wisconsin.data.txt", header=None, sep=',', prefix='H')
# data = data[data['H6'] != '?']
# data['H6'] = data['H6'].apply(pd.to_numeric)
# data = data.dropna()
# data = data.reset_index()
# labels = data.drop(columns=data.columns[:-1])
# data = data.drop(columns=data.columns[-1])
# #Drop first column because they are ID's
# data = data.drop(columns=data.columns[0])


# semente para o número aleatório
nseed = 100
random.seed(nseed)

directory_results = "forest_results_iono"
min_number_trees = 3
max_number_trees = 21

with open(directory_results+"/result.csv", 'w') as csv_result_file:
    writer = csv.writer(csv_result_file, delimiter=',')
    writer.writerow(['Number_Trees', 'Precision', 'Recall', 'F1_Score', 'Time'])

    for num_trees in range(min_number_trees, max_number_trees):
        print("============= Number of trees {} =============".format(num_trees))

        directory_num_tree = directory_results+"/numtree_{}".format(num_trees)
        if not os.path.exists(directory_num_tree):
            os.makedirs(directory_num_tree)

        directory_prefix = directory_num_tree+"/forest_folds_"
        k = 10
        kf = kfold.KFold()
        kf.make_kfold(data, labels, k, nseed)
        folds_precision = []
        folds_recall = []
        folds_f1 = []
        time_k_fold_begin = time()
        for i in range(k):
            print("======= Fold {} ======".format(i))
            fold_directory = directory_prefix+str(i)
            if not os.path.exists(fold_directory):
                os.makedirs(fold_directory)

            data_test, labels_test, data_train, labels_train = kf.get_data_test_train()
            kf.update_test_index()
            forest = RandomForest()
            start = time()
            forest.train(data_train, labels_train, nseed, num_trees=num_trees, max_depth=5)
            end = time()
            # Printing trees
            forest.print_forest_trees(graphviz=True, sufix=fold_directory+"/tree")


            predictions = forest.predict(data_test)
            macro_p = macro_precision(labels_test[labels_test.columns[0]].tolist(), predictions)
            macro_r = macro_recall(labels_test[labels_test.columns[0]].tolist(), predictions)
            macro_f1 = macro_f1_score(labels_test[labels_test.columns[0]].tolist(), predictions)
            folds_precision.append(macro_p)
            folds_recall.append(macro_r)
            folds_f1.append(macro_f1)

            results_text = ""
            results_text += "Ground truth: {}\n".format(labels_test[labels_test.columns[0]].tolist())
            results_text += "Predictions: {}\n".format(predictions)
            results_text += "Precision per class: {}\n".format(precision(labels_test[labels_test.columns[0]].tolist(), predictions))
            results_text += "Recall per class: {}\n".format(recall(labels_test[labels_test.columns[0]].tolist(), predictions))
            results_text += "F1 Scrore per class: {}\n".format(f1_score(labels_test[labels_test.columns[0]].tolist(), predictions))
            results_text += "Macro precision: {0:.2f}%\n".format(macro_p * 100)
            results_text += "Macro recall: {0:.2f}%\n".format(macro_r * 100)
            results_text += "Macro F1 Score: {0:.2f}%\n".format(macro_f1 * 100)
            results_text += "Time took: {} seconds\n".format(end-start)

            with open(fold_directory+"/result", "w") as text_file:
                text_file.write(results_text)

            with open(fold_directory+'/forest.pkl', 'wb') as output:
                pickle.dump(forest, output, pickle.HIGHEST_PROTOCOL)

            print(results_text)
        time_end_k_fold = time()
        with open(directory_num_tree+"/result", "w") as result_file:
            result_file.write("Folds precision: {}\n".format(folds_precision))
            result_file.write("Folds recall: {}\n".format(folds_recall))
            result_file.write("Folds f1 score: {}\n".format(folds_f1))

            result_file.write("Folds precision average: {0:.2f}%\n".format( (sum(folds_precision) / len(folds_precision)) *100 ))
            result_file.write("Folds recall average: {0:.2f}%\n".format( (sum(folds_recall) / len(folds_recall)) * 100 ))
            result_file.write("Folds f1 score average: {0:.2f}%\n".format( (sum(folds_f1) / len(folds_f1)) *100 ))
        writer.writerow([str(num_trees),
                         "{0:.2f}".format((sum(folds_precision) / len(folds_precision)) *100),
                         "{0:.2f}".format((sum(folds_recall) / len(folds_recall)) * 100),
                         "{0:.2f}".format((sum(folds_f1) / len(folds_f1)) *100),
                         "{}".format(time_end_k_fold-time_k_fold_begin)])