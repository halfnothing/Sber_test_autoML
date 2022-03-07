import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from metrics import *

def eval_train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        classifier,
        metric,
        test_size=0.25,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None
):
    """
    basic train/test-split test
    :param stratify:
    :param shuffle:
    :param train_size:
    :param random_state:
    :param X:
    :param y:
    :param classifier:
    :param test_size:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        train_size=train_size, shuffle=shuffle, stratify=stratify
    )
    classifier.fit(X_test, y_test)


def stratified_k_fold():
    scores = cross_val_score

class Validator:
    def __init__(
            self,
            method='train/test-split',
            metric='precision',
            test_size=0.4,
            **kwargs
    ):
        pass

    def __set_method__(self, method, **kwargs):
        if method == 'train/test-split':
            self.__method = train_test_split
        elif method == 'stratified-k-fold':
            self.__method = stratified_k_fold


