from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import pandas as pd


class SVMClassifier:
    def __init__(self, X_df, y_df, C, kernel, gamma):
        """
        Initializes the SVMClassifier with specified dataset and SVM parameters.

        :param X_df (DataFrame): The feature dataset used for training the SVM model.
        :param y_df (Series/DataFrame): The target labels corresponding to the features in X_df.
        :param C (float): Regularization parameter. 
        :param kernel (str): Specifies the kernel type to be used in the algorithm.
        :param gamma (str or float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        """
        self.X_df = X_df
        self.y_df = y_df
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.train_svm()

    def train_svm(self):
        """
        Trains the Support Vector Machine classifier using the initialized parameters.
        Prints the training parameters and indicates when the SVM has been trained.
        """
        #print('training params=', self.C, self.kernel, self.gamma)
        svm_clf = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, random_state=42)
        svm_clf.fit(self.X_df, self.y_df)
        self.model = svm_clf
        print("SVM trained")

    def predict(self, input_features):
        """
        Predicts the class labels for the provided feature set using the trained SVM model.

        :param input_features (array-like): A set of input features to classify.
        :return array: Predicted class labels for each input feature set.
        """
        return self.model.predict(input_features)

    def optimize(self):
        """
        Optimizes the parameters of the SVM model using downsampling and GridSearchCV.

        Performs downsampling to ensure balanced representation from each namespace before using grid search to find
        the optimal parameter combination. 

        :return dict: The best parameter combination found during the optimization.
        """
        print("Optimizing SVM")

        #print("Downsampling started")
        X_df_dup = self.X_df.copy()
        X_df_dup['namespace'] = self.dataset['namespace']
        unique_namespaces = X_df_dup["namespace"].unique()
        downsampled_dfs = []
        downsampled_labels = []
        n_samples_per_namespace = int(0.05 * len(X_df_dup) / len(unique_namespaces))

        # Downsampling process for each namespace
        for namespace in unique_namespaces:
            namespace_df = X_df_dup[X_df_dup["namespace"] == namespace]
            # Ensure not to sample more than available
            if len(namespace_df) < n_samples_per_namespace:
                print(
                    f"Warning: Not enough samples in namespace {namespace}. Using all available samples."
                )
                downsampled_df = namespace_df
            else:
                downsampled_df =  resample(
                    namespace_df,
                    replace=False,
                    n_samples=n_samples_per_namespace,
                    random_state=42,
                )
            downsampled_y = downsampled_df['namespace']
            downsampled_df = downsampled_df.drop('namespace', axis = 1)
            downsampled_dfs.append(downsampled_df)
            downsampled_labels.append(downsampled_y)
        # Resetting index to align X and y
        X_downsampled = pd.concat(downsampled_dfs).reset_index(drop=True)
        y_downsampled = pd.concat(downsampled_labels).reset_index(drop=True)
        #print("Downsampling done for optimization")

        #print("Grid search started for SVM")
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        }

        grid_search = GridSearchCV(
            SVC(random_state=42), param_grid, cv=5, scoring="accuracy"
        )
        grid_search.fit(X_downsampled, y_downsampled)

        print("Optimized SVM")

        return grid_search.best_params_
