# datasets
from datasets import *

# kNN Classifier
from kNNClassifier import *

# distances
from distances import *

# utils
from utils import *

def main_iris():

    """
    Description:
        Main function to train kNN models
    
    Parameters:
        None

    Returns:
        None    
    """

    # dataset hyperparameters 
    test_size = 0.2
    random_state = 42
    
    # kNN classifier hyperparameters
    k = 5
    metric = 'euclidean'
    resolution = 0.02

    # load dataset
    iris = Iris(test_size = test_size, random_state = random_state)
    # visualize
    iris.iris_visualize()
    # load train test split for training
    X_train, X_test, y_train, y_test = iris.load_iris()

    iris_features, iris_targets = iris.iris_metadata()

    print('---------------------------------------------------Dataset----------------------------------------------------')
    print('Loading Iris Dataset...')
    print('\nThe Features of the Iris Dataset are:', ', '.join(iris_features))
    print('\nThe Labels for the Iris Dataset are:', ', '.join(iris_targets))
    print(f'\nIris contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nk Nearest Neighbor Classifier\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Number of neighbors k =', k)
    print('Distance metric used is', metric)
    print('Training in progress...')

    # kNN Classifier
    kNN = kNNClassifier(k = k, metric = metric)
    # fit our train data
    kNN.fit(X = X_train, y = y_train)
    # predict on test
    predictions = kNN.predict(X_test)
    predictions = torch.tensor(predictions)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('Done Training!') 

    print('---------------------------------------------------Testing----------------------------------------------------')    
    print('kNN Accuracy = {:.2f}%'.format(acc))
    print('---------------------------------------------------Plotting---------------------------------------------------')
    print('Note: plotting will take some time, so please be patient')
    print('Plotting Iris Sepals...')
    # Iris Sepals
    X_train_sepals = X_train[:, [0, 1]]
    save_path_sepal = visualize_decision_boundaries_iris(X_train_reduced = X_train_sepals, y_train = y_train, metric = metric, 
                                                                                    resolution = resolution, sepal_or_petal = 'Sepal')
    print(f'Done, please refer to {save_path_sepal} to see sepal decision boundary')
    # Iris Petals
    print('Plotting Iris Petals...')
    X_train_petals = X_train[:, [2, 3]]
    save_path_petal = visualize_decision_boundaries_iris(X_train_reduced = X_train_petals, y_train = y_train, metric = metric, 
                                                                                    resolution = resolution, sepal_or_petal = 'Petal')
    print(f'Done, please refer to {save_path_petal} to see petal decision boundary')
    print('--------------------------------------------------------------------------------------------------------------')

if __name__ == '__main__':

    # runs iris dataset
    main_iris()
