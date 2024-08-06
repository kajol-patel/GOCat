from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier():
    def __init__(self, X_df, y_df, k):
        self.X_df = X_df
        self.y_df = y_df
        self.k = k
        self.train_knn()
    
    def train_knn(self):
        knn = KNeighborsClassifier(n_neighbors = self.k)
        knn.fit(self.X_df, self.y_df)
        self.model = knn
        
    def predict(self, input_features):
        self.model.predict(self.input_features)