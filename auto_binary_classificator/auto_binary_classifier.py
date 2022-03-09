import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# classifications models
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from .utils.validation import Validator
from sklearn.linear_model import LogisticRegression


class AutoBinaryClassifier:
    """
    Parameters
    ----------
    metric: str, callable metric
    normalization: bool, 'method', callable, list of normalization methods, default=False
        normalization == True then the features will be normalized.
        normalization == 'method' then the features will be normalized with using 'method'
        callable(normalization) == True then use function for normalization;
    category_in_data: bool default=False
    validator: Validator class that contain validation algorithm
    """
    def __init__(self, metric='precision',
                 normalization=False,
                 category_in_data=False,
                 validator=None
                 ):
        self.metric = metric
        self.normalization = normalization
        self.category_in_data = category_in_data
        if not validator:
            self.validator = Validator(metric=metric)
        elif isinstance(validator, Validator):
            self.validator = validator
        self.__set_classifiers()
        self.scores = np.zeros(len(self.classifiers))
        self.best_model = None

    def __set_classifiers(self):
        self.classifiers = [
            GaussianNB(),
            DecisionTreeClassifier(max_depth=10),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=10e2),
            KNeighborsClassifier(n_neighbors=5),
            LogisticRegression()
        ]

    def fit(self, X, y) -> None:
        """
        :param X: The data used to prediction
        :param y: target
        :return:
        """

        # validation
        scores = np.zeros(len(self.classifiers))
        for i, classifier in enumerate(self.classifiers):
            scores[i] = self.validator(X, y, classifier)

        self.best_model = self.classifiers[np.argmax(scores, axis=0)]
        self.scores = scores


    def predict(self, y) -> np.array:
        return self.best_model.predict(y)
