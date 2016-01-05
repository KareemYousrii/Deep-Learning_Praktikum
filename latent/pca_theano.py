# load data
from load_data import load_data

# theano
import theano
import theano.tensor.nlinalg as linalg
import theano.tensor as T

# scikit-learn pca
from sklearn import decomposition

# plotting
import matplotlib.pyplot as plt


class PCA(object):

    def __init__(self,  M):
        """
        :param M:   The new (reduced) dimension of the
                    data.
        """
        self.dim = M
        self.mean = None
        self.principal = None

    def transform(self, X):
        """
        Perform dimensionality reduction of the input matrix X

        :param X: The matrix of observations, where the training samples
        populate the rows, and the features populate the columns

        :return: Xtilde, the dimensionally reduced representation of the data
        """
        # center the data by subtracting the mean
        self.mean = T.mean(X, axis=0)
        X -= self.mean
        U, s, V = linalg.svd(X, full_matrices=False)

        # Keep track of the 'M' principal directions
        # The svd actually produces V^T, so the
        # principal directions are stored in the rows of
        # V as opposed to the columns
        self.principal = V[:self.dim]

        # Return the transformed data
        return linalg.dot(X, T.transpose(self.principal))

    def inverse_transform(self, X):
        """
        Perform an approximation of the input matrix of observations
        to the original dimensionality space

        :param X: The matrix of observations, where the training samples
        populate the rows, and the features populate the columns

        :return: Xhat, the dimensionality increased representation of the data
        """

        return linalg.dot(X, self.principal) + self.mean

if __name__ == '__main__':

    # load the MNIST data
    data = load_data()

    # Define the symbolic variables to be used
    X = T.matrix('X')
    Xtilde = T.matrix('Xtilde')

    # Initialize our model
    pca_ = PCA(100)

    # Theano function which fits the model to the
    # data i.e. applies dimensionality reduction
    transform = theano.function(
        inputs=[],
        outputs=pca_.transform(X),
        givens={
            X: data
        }
    )

    # Apply the dimensionality reduction
    reduced_data = transform()

    # Theano function which approximates the
    # given data to the original dimensionality
    # on which the model was trained
    approximate = theano.function(
        inputs=[],
        outputs=pca_.inverse_transform(Xtilde),
        givens={
            X: data,
            Xtilde: reduced_data
        }
    )

    Xhat = approximate()

    plt.matshow(Xhat[0,:].reshape((28,28)), cmap=plt.cm.gray)

    # compute PCA using scikit-learn for comparison
    pca = decomposition.PCA(n_components=100)
    pca.fit(data)
    X = pca.transform(data)
    X = pca.inverse_transform(X);
    plt.matshow(X[0,:].reshape((28,28)), cmap=plt.cm.gray)

    plt.show()