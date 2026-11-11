import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def average_accuracy(task_acc_matrix):
    """task_acc_matrix[j, k] = accuracy on task j after training task k."""
    mat = np.asarray(task_acc_matrix, dtype=float)
    if mat.size == 0:
        return 0.0
    T = mat.shape[1]
    return float(np.mean(mat[:, T - 1]))


def forgetting_measure(task_acc_matrix):
    """Mean forgetting over previous tasks after final task."""
    mat = np.asarray(task_acc_matrix, dtype=float)
    if mat.size == 0:
        return 0.0
    T = mat.shape[1]
    if T <= 1:
        return 0.0
    vals = []
    for j in range(T - 1):
        best_past = float(np.max(mat[j, :T - 1]))
        final = float(mat[j, T - 1])
        vals.append(max(0.0, best_past - final))
    return float(np.mean(vals)) if vals else 0.0


def backward_transfer(task_acc_matrix):
    mat = np.asarray(task_acc_matrix, dtype=float)
    T = mat.shape[1]
    if T <= 1:
        return 0.0
    vals = []
    for j in range(T - 1):
        vals.append(float(mat[j, T - 1] - mat[j, j]))
    return float(np.mean(vals)) if vals else 0.0
