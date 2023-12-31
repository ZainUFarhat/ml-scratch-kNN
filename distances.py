# numpy
import numpy as np
from numpy.linalg import norm

class Distnace():

    """
    Class to store all our distance functions
    """

    def __init__(self):
        """
        Description:
            Constructor for our Distance class

        Parameters:
            None
        
        Returns:
            None
        """
        pass

    # euclidean
    def euclidean_distance(self, x1, x2):

        """
        Description:
            Computes the euclidean distance between two data rows

        Parameters:
            x1: first data row
            x2: second data row

        Returns:
            euclidean
        """

        # compute the euclidean distance
        euclidean = np.sqrt(np.sum((x1 - x2) ** 2))

        # return
        return euclidean

    # manhattan
    def manhattan_distance(self, x1, x2):

        """
        Description:
            Computes the manhattan distance between two data rows

        Parameters:
            x1: first data row
            x2: second data row

        Returns:
            manhattan
        """

        # compute the manhattan distance
        manhattan = np.sum(np.abs(x1 - x2))

        # return
        return manhattan

    # minkowski
    def minkowski_distance(self, x1, x2, p):

        """
        Description:
            Computes the minkowski distance between two data rows

        Parameters:
            x1: first data row
            x2: second data row
            p: order of the norm

        Returns:
            minkowski
        """

        # compute the euclidean distance
        minkowski = np.sqrt(np.sum(x1 - x2) ** p)

        # return
        return minkowski

    # cosine
    def cosine_distance(self, x1, x2):

        """
        Description:
            Computes the cosine distance between two data rows
        
        Parameters:
            x1: first data row
            x2: second data row

        Returns:
            cosine_distance         
        """

        # compute the cosine similarity score
        cosine_similarity = np.dot(x1, x2)/(norm(x1) * norm(x2))

        # compute the cosine distance
        cosine_distance = 1 - cosine_similarity

        # return
        return cosine_distance
