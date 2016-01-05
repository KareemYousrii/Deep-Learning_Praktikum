"""
Also called "vector quantization", K-means can be viewed as a way of constructing
a `dictionary` D \in R^{n \times k} of k vectors, i.e. our K centroids, so that a
data vector x^{(i)} \in R^n, i = 1, ..., m can be mapped to a code vector that
minimizes the error in reconstruction.

The centroid index or cluster index is referred to as a "code", whereas the table
mapping codes to centroids and vice versa is referred to as a "code book"
"""
# Data loading and argument parsing
import cPickle
from optparse import OptionParser

# Receptive field visualization, taken from the deep learning tutorial
from utils import tile_raster_images

# Image processing
import skimage.transform
import skimage.color
from PIL import Image

# Theano and numpy
import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

def load_data():
    # load the data
    d = open('data_batch_1', 'rb')
    dict = cPickle.load(d)
    d.close()

    # load the class labels i.e. names
    l = open('batches.meta')
    label_names = cPickle.load(l)
    l.close()

    # Populate data arrays, where the rows represent the data features
    # and the columns represent the data samples
    X_train, y_train = dict['data'], np.asarray(dict['labels'])
    label_names = label_names['label_names']

    # Rescale the images and convert them into gray scale
    X_scaled = np.zeros((X_train.shape[0], 12*12))
    size = 12, 12
    for i in xrange(X_train.shape[0]):
        im = Image.fromarray(np.reshape(X_train[i], (32, 32, 3), order='F'))
        im.thumbnail(size)
        X_scaled[i] = \
            skimage.color.rgb2gray(skimage.transform.resize(np.reshape(X_train[i], (32, 32, 3), order='F'), (12, 12)))\
                .flatten()

    return theano.shared(value=np.transpose(X_scaled), name='X_train', borrow=True)


def Kmeans(X_train=None, K=300, epsilon_whitening=0.015):

    if X_train is None:
        X_train = T.matrix('X_train')

    ########################
    # Normalize the inputs #
    ########################

    # A constant added to the variance to avoid division by zero
    epsilon_norm = 10

    # We subtract from each training sample (each column in X_train) its mean
    X_train = X_train - T.mean(X_train, axis=0) / T.sqrt(T.var(X_train, axis=0) + epsilon_norm)

    #####################
    # Whiten the inputs #
    #####################

    sigma = T.dot(X_train, T.transpose(X_train)) / X_train.shape[1]
    U, s, V = linalg.svd(sigma, full_matrices=False)
    tmp = T.dot(U, T.diag(1/T.sqrt(s + epsilon_whitening)))
    tmp = T.dot(tmp, T.transpose(U))
    X_Whitened = T.dot(tmp, X_train)

    ######################
    # Training the Model #
    ######################

    # Initialization
    dimensions = X_Whitened.shape[0]
    samples = X_Whitened.shape[1]
    srng = RandomStreams(seed=234)

    # We initialize the centroids by sampling them from a normal
    # distribution, and then normalizing them to unit length
    # D \in R^{n \times k}
    D = srng.normal(size=(dimensions, K))
    D = D / T.sqrt(T.sum(T.sqr(D), axis=0))

    iterations = 30

    for i in xrange(iterations):

        # Initialize new point representations
        # for every pass of the algorithm
        S = T.zeros((K, samples))

        tmp = T.dot(D.T, X_Whitened)
        res = T.argmax(tmp, axis=0)
        max_values = tmp[res, T.arange(samples)]
        S = T.set_subtensor(S[res, T.arange(samples)], max_values)

        D = T.dot(X_Whitened, T.transpose(S))
        D = D / T.sqrt(T.sum(T.sqr(D), axis=0))

    return D

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-k", "--num_clusters", dest="num_clusters", default=300,
                  help="The number of clusters to use in kmeans")
    parser.add_option("-e", "--epsilon_whitening", dest="epsilon_whitening", default=0.015,
                  help="The epsilon to be used for ZCA whitening")

    (options, args) = parser.parse_args()

    X_train = load_data()
    X = T.matrix('X', dtype='float64')
    result = theano.function(
        inputs=[],
        outputs=Kmeans(X, K=options.num_clusters, epsilon_whitening=options.epsilon_whitening),
        givens={
            X: X_train
        }
    )

    D = result()

    image = Image.fromarray(
    tile_raster_images(X=np.transpose(D),
                           img_shape=(12, 12), tile_shape=(10, 30),
                           tile_spacing=(1, 1)))
    image.save('repflds.png')
