# **ml-scratch-kNN**
k Nearest Neighbor Algorithm

## **Description**
The following is my from scratch implementation of the k Nearest Neighbor algorithm.
    
### **Dataset**

For datasets I used the popular sklearn Iris Dataset.

First, let us visualize some of its feaures in two dimensional space.

Iris dataset contains four features - Sepals Length and Width, and Petals Length and Width.

It makes sense to plot the lengths and widths of the sepals and petals against each other.

**-** Sepal Visualization:

![alt text](https://github.com/ZainUFarhat/ml-scratch-kNN/blob/main/plots/iris/iris_sepal.png?raw=true)

**-** Petal Visualization:

![alt text](https://github.com/ZainUFarhat/ml-scratch-kNN/blob/main/plots/iris/iris_petal.png?raw=true)

### **Walkthrough**

**1.** Need the following packages installed: sklearn, numpy, collections, and matplotlib.

**2.** Once you made sure all these libraries are installed, eevrything is simple, just head to main.py and execute it.

**3.** Since code is modular, main.py can easily: \
\
    &emsp;**i.** Load the iris dataset \
    &emsp;**ii.** Split data into train and test sets \
    &emsp;**iii.** Build a kNN classifier \
    &emsp;**iv.** Fit the kNN classifer \
    &emsp;**v.** Predict on the test set \
    &emsp;**vi.** Plot the final decision boundary predictions

**4.** In main.py I specify a set of hyperparameters, these can be picked by the user. The main ones worth noting are number of neighbors (k) and the metric used (I provide metrics for euclidean, manhattan, minkowski, and cosine).

### **Results**

With hyperparameters k = 5 and metric = 'euclidean'. I was able to achieve 100% accuracy. The results can be shown from the decision boundaries. 

**Note**: I preferred to display the results for the standard euclidean metric. However, do feel free to try out different metrics (manhattan, minkowski, cosine) to understand and visualize different results.

Of course because of the high dimensionality, we must reduce our training space in order to visualize decision boundairs.

We will follow the same steps as above:

**-** Sepal Decision Boundary:

![alt text](https://github.com/ZainUFarhat/ml-scratch-kNN/blob/main/plots/iris/iris_decision_boundaries_sepal.png?raw=true)

**-** Petal Decision Boundary

![alt text](https://github.com/ZainUFarhat/ml-scratch-kNN/blob/main/plots/iris/iris_decision_boundaries_petal.png?raw=true)

As you can see all datapoints were correctly labelled, hence affirming our 100% accuracy.

Thank you for following my tutorial!