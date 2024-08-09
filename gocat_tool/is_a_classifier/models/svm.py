from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer, f1_score


class SVMClassifier:
    def __init__(self, X_df, y_df, C=5, kernel="rbf", gamma="scale"):
        self.X_df = X_df
        self.y_df = y_df
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.train_svm()

    def train_svm(self):
        print("training params=", self.C, self.kernel, self.gamma)
        svm_clf = OneVsRestClassifier(
            SVC(
                random_state=42,
                class_weight="balanced",
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
            )
        )
        svm_clf.fit(self.X_df, self.y_df)
        self.model = svm_clf
        print("SVM trained")

    def predict(self, input_features):
        return self.model.predict(input_features)

    def optimize(self):

        print("Optimizing SVM")

        svm_model = OneVsRestClassifier(SVC(random_state=42, class_weight="balanced"))
        param_grid = {
            "estimator__C": [0.1, 1, 10, 0.01, 0.5, 5, 50],
            "estimator__kernel": ["linear", "rbf", "poly"],
            "estimator__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1, 10],
        }
        scoring = {"F1-Score": make_scorer(f1_score, average="micro")}
        print("Grid search started for SVM")
        grid_search = GridSearchCV(
            svm_model, param_grid, scoring=scoring, refit="F1-Score", cv=5, n_jobs=-1
        )
        grid_search.fit(self.X_df, self.y_df)

        print("Optimized SVM")

        return grid_search.best_params_
