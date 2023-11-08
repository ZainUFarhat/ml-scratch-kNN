# torch
import torch

# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# kNNClassifier
from kNNClassifier import *

# Calculate accuracy - out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_true, y_pred):
  """
  calculates the accuracies of a given prediction

  Parameters:

    y_true: the true labels
    y_pred: our predicted labels
  
  Returns:

    accuracy
  """
  
  # find the number of correct predictions  
  correct = torch.eq(y_true, y_pred).sum().item()
  # calculate the accuracy
  acc = (correct/len(y_pred))*100
  # return the accuracy
  return acc 

def visualize_decision_boundaries_iris(X_train_reduced, y_train, metric, resolution, sepal_or_petal):

    """
    Description:
        Plots the decision bonudaries across the entire grid.
        This is to visualize our results for all possible datapoints on a given grid
    
    Parameters:
        X_train_reduced: 2D features selected from original X_train
        y_train: original y_train
        metric: distance function used
        resolution: resolution of grid for plotting the decision boundary
        sepal_or_petal: are these the sepal features or the petals?
    
    Returns:
        save_path
    """

    # hyperparameters
    k = 5

    # fit and train kNN classifier
    knn = kNNClassifier(k = k, metric = metric)
    knn.fit(X = X_train_reduced, y = y_train)

    # get the min and max limits for our x-axis and y-axis
    # here the two features chosen for X_train_reduced will compose the x-axis and y-axis respectively
    x_min, x_max = X_train_reduced[:, 0].min() - 1, X_train_reduced[:, 0].max() + 1
    y_min, y_max = X_train_reduced[:, 1].min() - 1, X_train_reduced[:, 1].max() + 1

    # we will create a meshgrid based on the x-axis and y-axis range
    # we will take resolution steps between min and max and we are aiming to classify all of these into our given labels
    xx, yy = np.meshgrid(np.arange(x_min.cpu(), x_max.cpu(), resolution), np.arange(y_min.cpu(), y_max.cpu(), resolution))

    # predict the labels for all points on the created meshgrid
    predictions = knn.predict(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))
    Z = np.array(predictions)

    # reshape so we can graph
    Z = Z.reshape(xx.shape)

    # create color maps for our plot, they correspond to the number of labels in iris
    # that is, first element is for label 0, second element is for label 1, and third element is for label 2
    cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # plot decision boundaries
    plt.figure(figsize = (7, 7))
    # use color mesh and specify the background
    plt.pcolormesh(xx, yy, Z, cmap = cmap_background)

    # scatter the two chosen features across each other and use y_train labels as colors based on the color map above
    plt.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c = y_train, cmap = cmap_bold)
    plt.title(f'Iris Decision Boundary Prediction - {sepal_or_petal}')
    plt.xlabel(f'{sepal_or_petal.lower()} length (cm)')
    plt.xlim(xx.min(), xx.max())
    plt.ylabel(f'{sepal_or_petal.lower()} width (cm)')
    plt.ylim(yy.min(), yy.max())

    save_path = f'plots/iris/iris_decision_boundaries_{sepal_or_petal.lower()}.png'

    # save plot
    plt.grid()
    plt.savefig(save_path)

    # return
    return save_path