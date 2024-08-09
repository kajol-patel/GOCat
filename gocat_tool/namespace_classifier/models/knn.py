from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class KNNClassifier():
    def __init__(self, X_df, y_df, k=5):
        self.X_df = X_df
        self.y_df = y_df
        self.k = k
        self.train_knn()
    
    def train_knn(self):
        knn = KNeighborsClassifier(n_neighbors = self.k)
        knn.fit(self.X_df, self.y_df)
        self.model = knn
        print('KNN fitted')
        
    def predict(self, input_features):
        return self.model.predict(input_features)
    
    def optimize(self):
        print('Optimizing KNN')
        X_train, X_test, y_train, y_test = train_test_split(self.X_df, self.y_df, test_size=0.2)
        max_k = int(len(y_train)**0.5)  # Setting max k to the square root of the training set size
        accuracy = []
        for i in range(1, max_k, 2):
            neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
            yhat = neigh.predict(X_test)
            accuracy.append(accuracy_score(y_test, yhat))
        optimized_k = accuracy.index(max(accuracy)) + 1
        print('Optimized KNN at k = ', str(optimized_k))
        return optimized_k