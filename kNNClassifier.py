# collections
from collections import Counter

# distances
from distances import *

# kNN Classifier
class kNNClassifier():

    """
    Description:
        My from scratch implementation of the k Nearest Neighbor Classifier
    """

    # constructor
    def __init__(self, k, metric, p = 3):

        """
        Description:
            Constructor for our kNNClassifier class

        Parameters:
            k: number of neighbors
            metric: our choice of a distance function
            p: norm value for minkowski distance, default is 3 (can be ignored)
        """

        # number of neighbors
        self.k = k
        self.p = p
        self.metric = metric
        self.distance = Distnace()

    
    # fit
    def fit(self, X, y):

        """
        Description:
            Fit our train set to estimator

        Parameters:
            X: our train features
            y: our train labels

        Returns:
            None
        """
        
        # create self references for training data, this is to use across all methods of our kNNClassifier class
        self.X_train, self.y_train = X, y

        # nothing to return
        return None

    # predict
    def predict(self, X):

        """
        Description:
            Predict the label of our input X

        Parameters:
            X: our data input

        Returns:
            predictions
        """

        # form predictions
        predictions = [self.votes(x) for x in X]

        # return these predictions
        return predictions

    # votes
    def votes(self, x):

        """
        Description:
            Majority vote for the most common labels

        Parameters:
            X: our data input
            
        Returns:
            label
        """

        # what distance measurement to use
        if self.metric == 'euclidean':
            # find all euclidean distances
            distances = [self.distance.euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.metric == 'manhattan':
            # find all manhattan distances
            distances = [self.distance.manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.metric == 'minkowski':
            # find all minkowski distances
            distances = [self.distance.minkowski_distance(x, x_train, p = self.p) for x_train in self.X_train]
        elif self.metric == 'cosine':
            # find all cosine distances
            distances = [self.distance.cosine_distance(x, x_train) for x_train in self.X_train]

        # get the closest k indices
        k_idx = torch.argsort(torch.tensor(distances))[:self.k]
        # get closest k labels
        k_labels = [self.y_train[i] for i in k_idx]

        # to determine our label we need to get a majority vote for all elements in k_labels
        # that is, the label that appears the most in k_labels is what gets chosen
        most_common = Counter(k_labels).most_common()
        # fetch our label
        label = most_common[0][0]

        # return
        return label