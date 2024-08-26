from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer, f1_score


class SVMClassifier:
    def __init__(self, X_df, y_df, C, kernel, gamma):
        """
        Initializes the SVMClassifier with specified dataset and SVM parameters within a OneVsRest framework for multi-class classification.

        :param X_df: The feature dataset used for training the SVM model.
        :param y_df: The target labels corresponding to the features in X_df.
        :param C: Regularization parameter. The strength of the regularization is inversely proportional to C.
        :param kernel: Specifies the kernel type to be used in the algorithm ('linear', 'rbf', 'poly').
        :param gamma : Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        """
        self.X_df = X_df
        self.y_df = y_df
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.train_svm()

    def train_svm(self):
        """
        Trains the Support Vector Machine classifier using the OneVsRest approach to allow handling of multi-class scenarios.
        Sets the class weight to 'balanced' to manage class imbalance by adjusting weights inversely proportional to class frequencies.
        """
        #print("training params=", self.C, self.kernel, self.gamma)
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
        #print("SVM trained")

    def predict(self, input_features):
        """
        Predicts the class labels for the provided feature set using the trained SVM model within the OneVsRest framework.

        :param input_features: A set of input features to classify.
        :return array: Predicted class labels for each input feature set.
        """
        return self.model.predict(input_features)

    def optimize(self):
        """
        Optimizes the parameters of the SVM model using GridSearchCV within the OneVsRest framework. 
        Utilizes the F1 micro-average score as the metric for evaluating model performance during parameter tuning.

        :return dict: The best parameter combination found during the optimization.
        """
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

        #print("Optimized SVM")

        return grid_search.best_params_
