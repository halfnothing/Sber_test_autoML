from typing import Callable
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss

from sklearn.metrics import log_loss

from sklearn.metrics import roc_auc_score


def get_metric(metric_name: str) -> Callable:
    """
    :param metric_name: str
    :return: Callable
        function to compute metric
    """
    metrics = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'top_k_accuracy': top_k_accuracy_score,
        'average_precision': average_precision_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
        'roc_auc': roc_auc_score,
        'log_loss': log_loss,
        'brier_score_loss': brier_score_loss
    }
    return metrics[metric_name]