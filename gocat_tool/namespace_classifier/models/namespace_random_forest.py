from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
import pandas as pd

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
        print('training params=', self.n_estimators, self.max_depth, self.min_samples_split, self.min_samples_leaf,
              self.bootstrap)

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

        print("Downsampling started")
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

        print("Downsampling done for optimization")

        param_grid = {
            "n_estimators": [30, 60, 100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }
        print("Grid search started for RF")
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring="accuracy",
        )
        grid_search.fit(X_downsampled, y_downsampled)

        print('optimized params=', grid_search.best_params_)
        return grid_search.best_params_
