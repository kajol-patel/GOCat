from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class KNNClassifier():
    def __init__(self, X_df, y_df, k):
        """
        Initializes the KNNClassifier with a DataFrame of features, labels, and a number of neighbors.

        :param X_df: Feature data for training the KNN classifier.
        :param y_df: Label data corresponding to the features in X_df.
        :param k: Number of neighbors to use.
        """
        self.X_df = X_df
        self.y_df = y_df
        self.k = k
        self.train_knn()
    
    def train_knn(self):
        """
        Trains the K-Nearest Neighbors (KNN) classifier on the provided feature set and labels.
        Initializes and fits a KNeighborsClassifier model with the number of neighbors specified in 'self.k'.
        """
        knn = KNeighborsClassifier(n_neighbors = self.k)
        knn.fit(self.X_df, self.y_df)
        self.model = knn
        print('Status: KNN fitted')
        
    def predict(self, input_features):
        """
        Predicts the class labels for the provided feature set using the trained KNN model.

        :param input_features: A set of input features to classify.
        :return array: Predicted class labels for each input feature set.
        """

        return self.model.predict(input_features)
    
    def optimize(self):
        """
        Optimizes the number of neighbors ('k') for the KNN classifier using cross-validation on the training data.
        Splits the dataset into training and testing subsets, then iteratively fits a KNN model with different values of 'k'.
        Evaluates each model's accuracy on the testing data to find the optimal 'k'.

        :return int: The optimized value of 'k' that yielded the highest accuracy on the testing set.
        """

        print('Status: Optimizing KNN')
        X_train, X_test, y_train, y_test = train_test_split(self.X_df, self.y_df, test_size=0.2)
        max_k = int(len(y_train)**0.5)  # Setting max k to the square root of the training set size
        accuracy = []
        for i in range(1, max_k, 2):
            neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
            yhat = neigh.predict(X_test)
            accuracy.append(accuracy_score(y_test, yhat))
        optimized_k = accuracy.index(max(accuracy)) + 1
        #print('Optimized KNN at k = ', str(optimized_k))
        return optimized_k