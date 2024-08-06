from sklearn.ensemble import RandomForestClassifier

class RFClassifier():
    def __init__(self, X_df, y_df, n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap):
        self.X_df = X_df
        self.y_df = y_df
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.train_rf()
    
    def train_rf(self):
        rf_clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth
                                        , min_samples_split = self.min_samples_split
                                        , min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap )
        rf_clf.fit(self.X_df, self.y_df)
        self.model = rf_clf
        
    def predict(self, input_features):
        self.model.predict(self.input_features)