"""One-vs-rest linear SVM with class-specific Naive Bayes feature weights.

Adapted method: Wang & Manning (2012), https://aclanthology.org/P12-2018/.
This implementation uses binary document counts for the ratios, TF-IDF inputs
for the SVM, and balanced binary class weights.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted


class NBSVM(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0, alpha=1.0, random_state=42):
        self.C = C
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        X = X.tocsr()
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        binary = X.copy()
        binary.data[:] = 1
        self.ratios_, self.estimators_ = [], []
        for label in self.classes_:
            target = y == label
            positive = np.asarray(binary[target].sum(axis=0)).ravel() + self.alpha
            negative = np.asarray(binary[~target].sum(axis=0)).ravel() + self.alpha
            ratio = np.log((positive / positive.sum()) / (negative / negative.sum()))
            estimator = LinearSVC(C=self.C, class_weight="balanced", dual="auto",
                                  max_iter=10000, random_state=self.random_state)
            estimator.fit(X.multiply(ratio).tocsr(), target)
            self.ratios_.append(ratio)
            self.estimators_.append(estimator)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        return np.column_stack([
            estimator.decision_function(X.multiply(ratio).tocsr())
            for ratio, estimator in zip(self.ratios_, self.estimators_)
        ])

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]
