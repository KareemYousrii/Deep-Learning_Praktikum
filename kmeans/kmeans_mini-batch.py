# Data loading and argument parsing
import cPickle
from optparse import OptionParser

# Receptive field visualization, taken from the deep learning tutorial
from utils import tile_raster_images

# Theano and numpy
import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

# Image processing
import skimage.transform
import skimage.color
from PIL import Image

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

    # return theano.shared(value=np.transpose(X_scaled), name='X_train', borrow=True)
    return X_scaled

def get_batch(data):
    return np.transpose(data[np.random.choice(10000, 1000, replace=False)])

class KmeansMiniBatch(object):

    def __init__(self, batch_size, data=None, K=300, epsilon_whitening=0.015):

        if data is None:
            self.X = T.matrix('X_train')
        else:
            self.X = data

        ########################
        # Normalize the inputs #
        ########################

        # A constant added to the variance to avoid division by zero
        self.epsilon_norm = 10
        self.epsilon_whitening = epsilon_whitening

        # We subtract from each training sample (each column in X_train) its mean
        self.X = self.X - T.mean(self.X, axis=0) / T.sqrt(T.var(self.X, axis=0) + self.epsilon_norm)

        #####################
        # Whiten the inputs #
        #####################

        sigma = T.dot(self.X, T.transpose(self.X)) / self.X.shape[1]
        U, s, V = linalg.svd(sigma, full_matrices=False)
        tmp = T.dot(U, T.diag(1/T.sqrt(s + self.epsilon_whitening)))
        tmp = T.dot(tmp, T.transpose(U))
        self.X = T.dot(tmp, self.X)

        ##################
        # Initialization #
        ##################
        self.K = K  # The number of clusters
        self.dimensions = self.X.shape[0]
        self.samples = batch_size
        self.srng = RandomStreams(seed=234)

        # We initialize the centroids by sampling them from a normal
        # distribution, and then normalizing them to unit length
        # D \in R^{n \times k}
        self.D = self.srng.normal(size=(self.dimensions, self.K))
        self.D = self.D / T.sqrt(T.sum(T.sqr(self.D), axis=0))

    def fit_once(self):

        # Initialize new point representations
        # for every pass of the algorithm
        S = T.zeros((self.K, self.samples))

        tmp = T.dot(self.D.T, self.X)
        res = T.argmax(tmp, axis=0)
        max_values = tmp[res, T.arange(self.samples)]
        S = T.set_subtensor(S[res, T.arange(self.samples)], max_values)

        self.D = T.dot(self.X, T.transpose(S))
        self.D = self.D / T.sqrt(T.sum(T.sqr(self.D), axis=0))

        return self.D

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-k", "--num_clusters", dest="num_clusters", default=300,
                  help="The number of clusters to use in kmeans")
    parser.add_option("-e", "--epsilon_whitening", dest="epsilon_whitening", default=0.015,
                  help="The epsilon to be used for ZCA whitening")

    (options, args) = parser.parse_args()

    X = T.matrix('X', dtype='float64')
    mini_batch = T.matrix('mini_batch', dtype='float64')

    kmeans = KmeansMiniBatch(
        batch_size=1000,
        data=X,
        K=options.num_clusters,
        epsilon_whitening=options.epsilon_whitening
    )

    func = theano.function(
        inputs=[mini_batch],
        outputs=kmeans.fit_once(),
        givens={
            X: mini_batch
        },
    )

    data = load_data()
    D= None
    for i in xrange(30):
        D = func(get_batch(data))

    image = Image.fromarray(
    tile_raster_images(X=np.transpose(D),
                           img_shape=(12, 12), tile_shape=(10, 30),
                           tile_spacing=(1, 1)))
    image.save('repflds_mini-batch.png')