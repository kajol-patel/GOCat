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
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        bootstrap,
    ):
        """
        Initializes the RandomForestClassifier with specified parameters and dataset.

        :param X_df: The feature dataset used for training the RandomForest model.
        :param y_df: The target labels corresponding to the features in X_df.
        :param n_estimators: The number of trees in the forest.
        :param max_depth: The maximum depth of the trees.
        :param min_samples_split: The minimum number of samples required to split an internal node.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        :param bootstrap: Whether bootstrap samples are used when building trees.
        """
        self.X_df = X_df
        self.y_df = y_df
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.train_rf()

    def train_rf(self):
        """
        Trains the RandomForest classifier using the initialized parameters. 
        Sets the class weight to 'balanced' to handle class imbalance by adjusting weights inversely proportional to class frequencies.
        """
        #print("training params=",self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.bootstrap,)

        rf_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            class_weight='balanced',random_state=42
        )
        rf_clf.fit(self.X_df, self.y_df)
        self.model = rf_clf
        #print("RF trained")

    def predict(self, input_features):
        """
        Predicts the class labels for the provided feature set using the trained RandomForest model.

        :param input_features: A set of input features to classify.
        :return array: Predicted class labels for each input feature set.
        """
        return self.model.predict(input_features)

    def optimize(self):
        """
        Optimizes the parameters of the RandomForest model using GridSearchCV with specified parameter ranges.
        Utilizes the F1 micro-average score to find the best model, focusing on balancing recall and precision, especially in multi-class settings.

        :return dict: The best parameter combination found during the optimization.
        """
        print("Optimizing RF")

        rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
        param_grid = {
            "n_estimators": list(range(20, 300, 50)),  
            "max_depth": list(range(3, 60, 3)),  
            "min_samples_split": [2, 5, 10, 20, 30],  
            "min_samples_leaf": [1, 2, 4, 10],  
        }

        scoring = {
            "F1-Score": make_scorer(f1_score, average="micro", needs_proba=False)
        }
        #print("Grid search started for RF")
        grid_search = GridSearchCV(
            rf_model, param_grid, cv=5, scoring=scoring, refit="F1-Score", n_jobs=-1
        )
        grid_search.fit(self.X_df, self.y_df)

        #print("optimized params=", grid_search.best_params_)
        return grid_search.best_params_
