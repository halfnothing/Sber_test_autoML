import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# classifications models
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.validation import Validator
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
                 validator=Validator
                 ):
        self.metric = metric
        self.normalization = normalization
        self.category_in_data = category_in_data
        self.validator = validator(metric)



    def fit(self, X, y) -> None:
        """
        :param X: The data used to prediction
        :param y: target
        :return:
        """

        classifiers = [
            GaussianNB(),
            DecisionTreeClassifier(max_depth=10),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=10e2),
            KNeighborsClassifier(n_neighbors=5),
            LogisticRegression()
        ]

        # validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        for classifier in classifiers:
            pass



