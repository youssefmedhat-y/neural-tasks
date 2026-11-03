import numpy as np


def compute_confusion_matrix(y_true, y_pred):
    matrix = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def compute_accuracy(y_true, y_pred):
    correct = 0
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == pred_val:
            correct += 1

    total = len(y_true)
    if total == 0:
        return 0.0
    return (correct / total) * 100.0


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y = np.asarray(y)
    if y.shape != y_pred.shape or len(y) == 0:
        raise ValueError('Expected one class label for each sample.')
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError('Class labels must be 0, 1, or 2.')
    matrix = compute_confusion_matrix(y, y_pred)
    return {
        'accuracy': compute_accuracy(y, y_pred),
        'confusion_matrix': matrix.tolist(),
        'predictions': y_pred.tolist(),
    }
