import numpy as np
from sklearn.datasets import make_classification
from auto_binary_classificator import AutoBinaryClassifier
from auto_binary_classificator.utils.validation import Validator


n_features = 10
X, y = make_classification(n_samples=100,
                           n_features=n_features,
                           n_informative=4,
                           n_classes=2,
                           shift=np.random.rand(n_features) * 10,
                           scale=np.random.rand(n_features) * 10)

auto_classifier_1 = AutoBinaryClassifier()
auto_classifier_1.fit(X, y)

validator = Validator(method='stratified-k-fold')
auto_classifier_2 = AutoBinaryClassifier(metric='accuracy', validator=validator)
auto_classifier_2.fit(X, y)

