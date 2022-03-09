import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from .metrics import get_metric


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
) -> float:
    """
    basic train/test-split test with using metric
    :param X:
    :param y:
    :param classifier:
    :param metric:
    :param stratify:
    :param shuffle:
    :param train_size:
    :param random_state:
    :param test_size:
    :return: float
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        train_size=train_size, shuffle=shuffle, stratify=stratify
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return metric(y_test, y_pred)


def stratified_k_fold(
        X: np.ndarray,
        y: np.ndarray,
        classifier,
        metric,
        n_splits=5,
) -> float:
    """
    :param X:
    :param y:
    :param classifier:
    :param metric:
    :param n_splits:
    :return: float
    """
    skf = StratifiedKFold(n_splits=n_splits)
    scores = []
    for train_index, test_index in skf.split(X, y):
        classifier.fit(X[train_index], y[train_index])
        y_pred = classifier.predict(X[test_index])
        scores.append(metric(y[test_index], y_pred))
    return np.mean(scores, axis=0)


class Validator:
    def __init__(
            self,
            method='train/test-split',
            metric='precision',
            **kwargs
    ):
        self.metric = metric
        self.method = method
        self.__set_validation_method__(method)

    def __set_validation_method__(self, method):
        """
        :param method: str
        :param kwargs: parameters for using validation method
        :return:
        """
        if method == 'train/test-split':
            self.__method = eval_train_test_split
        elif method == 'stratified-k-fold':
            self.__method = stratified_k_fold
        elif callable(method):
            self.__method = method

    def __call__(self, X, y, classifier):
        return self.__method(X, y, classifier, get_metric(self.metric))

