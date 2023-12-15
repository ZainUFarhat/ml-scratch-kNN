# sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

# visualization
import matplotlib.pyplot as plt

class Iris():

    """
    Class that holds sklearn Iris Dataset
    """

    # constructor
    def __init__(self, test_size, random_state):

        """
        Description:
            Initialize user controlled hyperparameters for datasets
        
        Parameters:
            test_size: percantage of data allocated for test
            random_state: set the seed for random generator
        
        Return:
            None
        """

        # test size and random state
        self.test_size = test_size
        self.random_state = random_state
        
    def iris_visualize(self):
        """
        Decsription:
            Visualize Iris datset from its metadata
        
        Parameters:
            None
        
        Returns:
            None
        """

        # load iris
        iris = datasets.load_iris()

        # feature data
        sepal_lengths, sepal_widths = iris.data[:, 0], iris.data[:, 1]
        petal_lengths, petal_widths = iris.data[:, 2], iris.data[:, 3]
        # targets
        targets = iris.target_names
        # corresponding color (which is also their name)
        colors = iris.target
        # the string title of sepal length and width features
        sepal_length_name, sepal_width_name = iris.feature_names[0], iris.feature_names[1]
        petal_length_name, petal_width_name = iris.feature_names[2], iris.feature_names[3]

        # set figure size
        plt.figure(figsize=(7, 7))

        # set background color to lavender
        ax = plt.axes()
        ax.set_facecolor("lavender")

        # scatterplot
        sc = plt.scatter(sepal_lengths, sepal_widths, c = colors)
        plt.title('Iris Dataset Scatterplot - Sepal')
        plt.xlabel(sepal_length_name)
        plt.ylabel(sepal_width_name)
        plt.legend(sc.legend_elements()[0], targets, loc = 'lower right', title = 'Classes')
        plt.grid()
        plt.savefig(f'plots/iris/iris_sepal.png')

        # set figure size
        plt.figure(figsize=(7, 7))
        
        # set background color to lavender
        ax = plt.axes()
        ax.set_facecolor("lavender")

        # scatterplot
        sc = plt.scatter(petal_lengths, petal_widths, c = colors)
        plt.title('Iris Dataset Scatterplot - Petal')
        plt.xlabel(petal_length_name)
        plt.ylabel(petal_width_name)
        plt.legend(sc.legend_elements()[0], targets, loc = 'lower right', title = 'Classes')
        plt.grid()
        plt.savefig(f'plots/iris/iris_petal.png')

        # nothing to return, we just want to save plots
        return None
    
    def iris_metadata(self):

        """
        Description:
            Fetches iris metadate
        
        Parameters:
            None
        
        Returns:
            feature_names, num_classes
        """

        # load iris
        iris = datasets.load_iris()

        # just fetch the feature names and target names
        sepal_length_name, sepal_width_name = iris.feature_names[0], iris.feature_names[1]
        petal_length_name, petal_width_name = iris.feature_names[2], iris.feature_names[3]
        targets = iris.target_names

        # return
        return [sepal_length_name, sepal_width_name, petal_length_name, petal_width_name], targets

    def load_iris(self):
    
        """
        Description:
            Load sklearn iris dataset
        
        Parameters:
            None
        
        Return:
            X_train, X_test, y_train, y_test
        """

        # load iris dataset
        iris = datasets.load_iris()

        # fetch features and labels
        X, y = iris.data, iris.target

        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)

        # return
        return X_train, X_test, y_train, y_test