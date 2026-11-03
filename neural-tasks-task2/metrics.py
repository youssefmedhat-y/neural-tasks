def compute_confusion_matrix(y_true, y_pred, class1_name="Class 1", class2_name="Class 2"):
    # Class 1 is positive (+1), Class 2 is negative (-1)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == 1 and pred_val == 1:
            TP += 1
        elif true_val == -1 and pred_val == 1:
            FP += 1
        elif true_val == -1 and pred_val == -1:
            TN += 1
        elif true_val == 1 and pred_val == -1:
            FN += 1

    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'class1_name': class1_name,
        'class2_name': class2_name,
    }


def compute_accuracy(y_true, y_pred):
    correct = 0
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == pred_val:
            correct += 1

    total = len(y_true)
    if total == 0:
        return 0.0
    return (correct / total) * 100.0
