import csv
import matplotlib.pyplot as plt

resource_paths = ["forest_results_breast/", "forest_results/",
                  "forest_results_iono/"]

for resource_files in resource_paths:
    with open(resource_files+'result.csv', 'r') as csv_result_file:
        spamreader = csv.DictReader(csv_result_file)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        num_trees = []
        for row in spamreader:
            precision_scores.append(float(row['Precision']))
            recall_scores.append(float(row['Recall']))
            f1_scores.append(float(row['F1_Score']))
            num_trees.append(int(row['Number_Trees']))

        plt.xlabel("Número de Árvores")
        plt.ylabel("F1 Score")
        plt.plot(num_trees, f1_scores)
        plt.axis([min(num_trees), max(num_trees), min(f1_scores)-5, 100])
        plt.xticks(num_trees)
        plt.savefig(resource_files+"f1_score_grow.png", format="png")
        plt.clf()

        plt.xlabel("Número de Árvores")
        plt.ylabel("Precisão")
        plt.plot(num_trees, precision_scores)
        plt.axis([min(num_trees), max(num_trees), min(precision_scores) - 5, 100])
        plt.xticks(num_trees)
        plt.savefig(resource_files + "precision_grow.png", format="png")
        plt.clf()

        plt.xlabel("Número de Árvores")
        plt.ylabel("Recall")
        plt.plot(num_trees, recall_scores)
        plt.axis([min(num_trees), max(num_trees), min(recall_scores) - 5, 100])
        plt.xticks(num_trees)
        plt.savefig(resource_files + "recall_grow.png", format="png")
        plt.clf()