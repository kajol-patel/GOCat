from sklearn.svm import SVC


class SVMClassifier():
    def __init__(self, X_df, y_df, C, kernel, gamma):
        self.X_df = X_df
        self.y_df = y_df
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.train_svm()
    
    def train_knn(self):
        svm_clf = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)
        svm_clf.fit(self.X_df, self.y_df)
        self.model = svm_clf
        
    def predict(self, input_features):
        self.model.predict(self.input_features)