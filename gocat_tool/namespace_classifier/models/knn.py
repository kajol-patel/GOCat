class KNNClassifier():
    def __init__(self, X_df, y_df, k):
        self.X_df = X_df
        self.y_df = y_df
        self.k = k