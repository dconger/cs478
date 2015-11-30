import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.cross_validation import KFold


class LearnerSuite(object):

    def __init__(self, tuning_ranges=None, models=None, cv=None, njobs=1, verbose=False):
        self.scorer = make_scorer(accuracy_score)
        self.tuning_ranges = tuning_ranges
        self.models = models
        self.model_names = []

        for model in models:
            self.model_names.append(model.__class__.__name__)
            if model.__class__.__name__ not in tuning_ranges:
                raise ValueError('No tuning parameters for', model.__class__.__name__)

        self.verbose = verbose
        self.cv = cv
        self.best_scores = dict()
        self.best_models = dict()
        self.njobs = njobs

    def fit(self, X_train, y_train):
        # Nested Cross-validation
        self.cv = KFold(len(y_train), n_folds=self.cv)

        for i in range(len(self.models)):
            best_estimator, best_score, best_params = self.cross_validate(X_train, i, y_train)

            self.models[i] = best_estimator
            self.best_models[self.model_names[i]] = best_estimator
            self.best_scores[self.model_names[i]] = best_score

    def cross_validate(self, X, model_idx, y):

        gs = GridSearchCV(self.models[model_idx], self.tuning_ranges[self.model_names[model_idx]], 
                                                scoring=self.scorer, n_jobs=self.njobs, cv=self.cv)
        gs.fit(X, y)

        return gs.best_estimator_, gs.best_score_, gs.best_params_

    def score_all(self, X_test, y_test):
        y_scores = {name: model.score(X_test, y_test) for name, model in zip(self.model_names, self.models)}

        return y_scores

    def predict_all(self, X_test):
        y_predict_all = {name: model.predict(X_test) for name, model in zip(self.model_names, self.models)}

        return y_predict_all
