import cPickle
import gzip
import os

import numpy
import theano
import theano.tensor as T
from matplotlib import pyplot
import matplotlib

from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy

        data_x = data_x
        data_y = data_y

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    # test_set_x, test_set_y = shared_dataset(test_set)
    # valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y)]
    return rval

class sA(object):
    def __init__(self, numpy_rng, input=None, n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        """
        Initialize the sparse Auto-encoder class by specifying the number of
        visible units, the dimensionality of the input, the number of hidden
        units, the dimensionality of the code.

        :param numpy_rng: random number generator used for weight initialization
        :param input: a symbolic description of the input
        :param n_visible: number of visible units
        :param n_hidden: number of hidden units
        :param W: The set of weights #check
        :param bhid: a set of bias values for the hidden units
        :param bvis: a set of bias values for the visible units
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not W:
            # W is uniformely sampled from -4*sqrt(6./(n_visible+n_hidden))
            # and 4*sqrt(6./(n_hidden+n_visible))
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

            self.W = W

            # b corresponds to the bias of the hidden layer
            self.b = bhid

            # b_prime corresponds to the bias of the visible layer
            self.b_prime = bvis

            # We use tied weights i.e. the weights for the decoder are
            # the transpose of the encoder weights
            self.W_prime = self.W.T

            # We use a symbolic input so that later on, a deep auto-encoder
            # can be created by stacking layers of auto-encoders, in which case,
            # the input to one auto-encoder is the output of the one beneath it.
            if input is None:
                self.x = T.dmatrix(name='input')
            else:
                self.x = input

            # Why is W_prime not included in the params? Is it because
            # if we optimize W, we do not need to optimize the transpose
            # of W? perhaps...
            self.params = [self.W, self.b, self.b_prime] #check

            # Square of L2 norm; one regularization option is to enforce the
            # square of L2 norm to be small
            self.L2_sqr = (
                (self.W ** 2).sum()
                + (self.W_prime ** 2).sum()
            )

            # A symbolic variable for the average of the hidden activation units
            # which for the sparsity cost
            self.avg_hidden = T.vector("avg_hidden")


    def get_hidden_values(self, input):
        """
        Computes the values of the hidden layer

        y = s(W.x + b)
        """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_output(self, hidden):
        """
        Computes the reconstructed input given the `code`

        z = s(W_prime.y + b_prime)
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, learning_rate, L2_reg=0.001, beta=0.1, sparsity_param=0.05):
        """
        This function computes the cost and updates for one training step
        """
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_output(y)

        # note: we sum over the size of a data point i.e. the dimensionality
        # of a training example; since we are using mini-batches, L will be
        # a vector, with one entry per training point in the mini-batch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1)

        sparsity_cost = T.sum(sparsity_param * T.log(sparsity_param/self.avg_hidden) + (1-sparsity_param) * T.log(1 - sparsity_param/1 - self.avg_hidden))

        # We compute the cost over the mini-batch by averaging the costs
        # obtained for all the training samples in the mini-batch
        cost = T.mean(L) #+ (beta * sparsity_cost) #+ (L2_reg * self.L2_sqr)

        # Compute the gradients of the cost with resepct to the parameters
        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

def test_sA(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='sA_plots'):
    """
    :param learning_rate: learning rate used to update the parameters
    of the sparse auto-encoder during training
    :param training_epochs: Number of training epochs
    :param dataset: The dataset to be used for training
    :param batch_size: the size of each batch of training examples
    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # Compute the number of mini-batches for training
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # Allocate symbolic variables for the data
    index = T.lscalar() # index used to denote the mini-batch to be used
    x = T.matrix('x')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ######################
    # BUILDING THE MODEL #
    ######################

    rng = numpy.random.RandomState(123)

    sa = sA(
        numpy_rng=rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    # produce the mean over all columns i.e. the average of activations for
    # each hidden unit
    avg_hidden_activations = T.mean(sa.get_hidden_values(train_set_x), axis=0)

    cost, updates = sa.get_cost_updates(learning_rate=learning_rate)

    train_sa = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            sa.avg_hidden: avg_hidden_activations
        }
    )

    ############
    # TRAINING #
    ############

    for epoch in xrange(training_epochs):
        # holds the cost for each of the mini-batches using per training epoch
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_sa(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    image = Image.fromarray(
    tile_raster_images(X=sa.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(25, 20),
                           tile_spacing=(1, 1)))
    image.save('repflds_no_sparse.png')

if __name__ == '__main__':
    test_sA()
