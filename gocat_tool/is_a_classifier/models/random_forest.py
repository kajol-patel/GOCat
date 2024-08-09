from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
import pandas as pd
from sklearn.metrics import make_scorer, f1_score


class RFClassifier:
    def __init__(
        self,
        X_df,
        y_df,
        n_estimators=60,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
    ):
        self.X_df = X_df
        self.y_df = y_df
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.train_rf()

    def train_rf(self):
        
        print(
            "training params=",
            self.n_estimators,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.bootstrap,
        )

        rf_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
        )
        rf_clf.fit(self.X_df, self.y_df)
        self.model = rf_clf
        print("RF trained")

    def predict(self, input_features):
        return self.model.predict(input_features)

    def optimize(self):

        print("Optimizing RF")

        rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
        param_grid = {
            "n_estimators": list(range(20, 300, 50)),  # Number of trees in the forest
            "max_depth": list(range(3, 60, 3)),  # Maximum depth of the tree
            "min_samples_split": [
                2,
                5,
                10,
                20,
                30,
            ],  # Minimum number of samples required to split an internal node
            "min_samples_leaf": [
                1,
                2,
                4,
                10,
            ],  # Minimum number of samples required to be at a leaf node
        }

        scoring = {
            "F1-Score": make_scorer(f1_score, average="micro", needs_proba=False)
        }
        print("Grid search started for RF")
        grid_search = GridSearchCV(
            rf_model, param_grid, cv=5, scoring=scoring, refit="F1-Score", n_jobs=-1
        )
        grid_search.fit(self.X_df, self.y_df)

        print("optimized params=", grid_search.best_params_)
        return grid_search.best_params_
