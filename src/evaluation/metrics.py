import numpy as np


def get_metrics(y_pred: list[str], y_true: list[str]) -> dict[str, float]:
    # accuracy
    accuracy = np.mean([pred == true for pred, true in zip(y_pred, y_true)])
    
    return {
        "accuracy": accuracy
    }
