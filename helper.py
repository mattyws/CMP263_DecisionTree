import collections


def recall(true, predicted):
    hit = [x for x, y in zip(true, predicted) if x == y]
    true_values_count = collections.Counter(true)
    hit_count = collections.Counter(hit)
    recalls = {}
    for key in true_values_count.keys():
        if key in hit_count.keys():
            recalls[key] = hit_count[key] / true_values_count[key]
        else:
            recalls[key] = 0
    return recalls

def precision(true, predicted):
    hit = [x for x, y in zip(true, predicted) if x == y]
    predicted_values_count = collections.Counter(predicted)
    hit_count = collections.Counter(hit)
    precisions = {}
    for key in predicted_values_count.keys():
        if key in hit_count.keys():
            precisions[key] = hit_count[key] / predicted_values_count[key]
        else :
            precisions[key] = 0
    return precisions

def macro_recall(true, predicted):
    recalls = recall(true, predicted)
    macro_recall = 0
    for key in recalls.keys():
        macro_recall += recalls[key]
    macro_recall /= len(recalls.keys())
    return macro_recall

def macro_precision(true, predicted):
    precisions = precision(true, predicted)
    macro_precision = 0
    for key in precisions.keys():
        macro_precision += precisions[key]
    macro_precision /= len(precisions.keys())
    return macro_precision

def f1_score(true, predicted):
    precisions = precision(true, predicted)
    recalls = recall(true, predicted)
    f1_scores = {}
    for key in precisions.keys():
        if precisions[key] == 0 and recalls[key] == 0:
            f1_scores[key] = 0
        else:
            f1_scores[key] = 2 * ( (precisions[key]*recalls[key]) / (precisions[key]) + recalls[key] )
    return f1_scores

def macro_f1_score(true, predicted):
    macro_p = macro_precision(true, predicted)
    macro_r = macro_recall(true, predicted)
    return 2 * ((macro_p * macro_r)/(macro_p + macro_r))