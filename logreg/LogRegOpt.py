import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
import climin

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

class LogisticRegression(object):
    # Classification is done by projecting data points onto a set
    # of hyperplanes, the distance to which is used to determine
    # a class membership probability.

    def __init__(self, input, n_in, n_out):
        """Initialize the parameters of logistic Regression

        :type input: theano,tensor.TensorType
        :param input: symbolic variable that describes
            the input of the architecture.

        :type n_in: int
        :param n_in: number of input units, the dimension
            of space in which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension
            of space in which the labels lie
        """

        # Initialize theata = (W,b) with 0s; W gets the shape
        # (n_in, n_out), while b is a vector of n_out elements,
        # making theta a vector of n_in*n_out + n_out elements
        self.theta = theano.shared(
            value=numpy.zeros(
                n_in * n_out + n_out,
                dtype=theano.config.floatX
            ),
            name='theta',
            borrow=True
        )

        # W is represented by the first n_in * n_out elements of
        # theta
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))

        # b is represented by the last n_out elements
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]

        # This is analogous to computing P(C_k|\phi): THe posterior probability
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # Choose the class with the highest probability as the predicted class
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y, lmda):
        """ Returns the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: A vector of labels corresponding to the training samples

        Even though the loss is formally defined as the sum
        over training samples errors, in practice using the mean
        allows for the learning rate to be less dependent of the
        minibatch size.

        """

        # 1. T.log(self.p_y_given_x) is a matrix of log probabilities
        # (call it LP) with one row per example, and one column per
        # class
        #
        # 2. LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]]

        return -(T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])) + T.mean(lmda * T.square(self.W))

    def errors(self, y):
        """ Returns a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch

        :type y: theano.tensor.TensorType
        :param y: A vector of labels corresponding to the training samples
        """

        # check if y has the same dimensions as y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # Check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # The T.neq operator returns a vector of 0s and 1s,
            # where 1 represents a mistake in the prediction
            # i.e. Zero-one loss
            return T.mean(T.neq(self.y_pred, y))

        else:
            return NotImplementedError()


def load_data(dataset):
    """ Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    """

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == " " and not os.path.isfile(dataset):
        # Check if dataset is in the data directory
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

    # def transform_training(train_set):
    #     rng = numpy.random.RandomState()
    #     for image in train_set[0:50000, :]:
    #         im = Image.fromarray(image * 255.)
    #         train_set.append(im.rotate(rng.randint(0, 360)))


    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables.
        The reason for doing this is performance. If the data were
        not to be stored in shared variables, the minibatches would
        be copied on request, resulting in a huge performance degredation.
        Whereas, if you use theano shared variables, theano could copy the
        entire data to the GPU in a single call when the shared variables
        are constructed. Afterwards, the GPU can access any minibatch by
        taking a slice from the shared variables, without any copying necessary.
        """

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX), borrow=borrow)

        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX), borrow=borrow)

        # When storing data in the GPU it has to be stored as floats
        # However, Since the labels are used as indices, they need
        # to be cast into ints before they can be used.
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

def custom_optimization_mnist(n_epochs=200, dataset='mnist.pkl.gz', optimization='nlcg'):
    """
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the pickled MNIST dataset file
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    batch_size = 600

    # Compute the number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    n_in = 28 * 28  # number of input units
    n_out = 10  # number of output units

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # index to minibatch
    index = T.lscalar()

    # x and y represent a minibatch
    x = T.matrix('x')
    y = T.ivector('y')

    # construct the logistic regression classifier
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # The cost we minimize during the training is the negative
    # log likelihood of the model (in symoblic format?)
    cost = classifier.negative_log_likelihood(y, 0.3).mean() #back

    # The 'givens' parameter allows us to separate the description
    # of the model from the exact definition of the input variables.
    # Namely, the 'givens' parameter modifies the graph, by substituting
    # the keys with the associated values. Above, we used normal Theano
    # variables to build the model, which were then substituted by
    # shared variables holding the dataset on the GPU.

    # compiling a Theano function the mean of the zero-one loss
    # function by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index : index + batch_size],
            y: test_set_y[index : index + batch_size]
        },
        name="test"
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index : index + batch_size],
            y: valid_set_y[index : index + batch_size]
        },
        name="validate"
    )

    training_loss = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index : index + batch_size],
            y: train_set_y[index : index + batch_size]
        },
        name="training_loss"
    )

    # compile a theano function that returns the cost of a minibatch
    batch_cost = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index: index + batch_size],
            y: train_set_y[index: index + batch_size]
        },
        name="batch_cost"
    )

    # compile a theano function that returns the gradient of a
    # minibatch with respect to theta
    batch_grad = theano.function(
        inputs=[index],
        outputs=T.grad(cost, classifier.theta),
        givens={
            x: train_set_x[index: index + batch_size],
            y: train_set_y[index: index + batch_size]
        },
        name="batch_grad"
    )

    # create a function which computes the average cost on the
    # training set
    def train_fn(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        # An array containing the cost for each of the minibatches
        train_losses = [batch_cost(i * batch_size)
                        for i in xrange(n_train_batches)]
        return numpy.mean(train_losses)

    # create a function that computes the average gradient of
    # the cost w.r.t theta
    def train_fn_grad(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        grad = batch_grad(0)
        for i in xrange(1, n_train_batches):
            grad += batch_grad(i * batch_size)
        return grad/n_train_batches


    # early-stopping parameters
    patience = [5000, 2]
    improvement_threshold = 0.995

    # Keeping track of training, testing and validation errors
    # per epoch
    validations = []
    tests = []
    trainings = []

    validation_scores = [numpy.inf, 0, 0, None]

    # creates the validation function
    def callback(theta_value, current_epoch):
        classifier.theta.set_value(theta_value, borrow=True)

        # compute the validation loss
        validation_losses = [validate_model(i * batch_size)
                             for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)
        validations.append(this_validation_loss * 100.)

        # compute the test loss
        test_losses = [test_model(i*batch_size)
                       for i in xrange(n_test_batches)]
        this_test_loss = numpy.mean(test_losses)
        tests.append(this_test_loss * 100.)

        # compute the training loss
        training_losses = [training_loss(i * batch_size)
                           for i in xrange(n_train_batches)]
        this_train_loss = numpy.mean(training_losses)
        trainings.append(this_train_loss * 100.)

        print('validation error %f %%' % (this_validation_loss * 100.,))

        # check if the new validation error is better than our current
        # validation score
        if this_validation_loss < validation_scores[0]:
            
            # improve patience if loss improvement is good enough
            if this_validation_loss < validation_scores[0] * \
                    improvement_threshold:
                patience[0] = max(patience[0], current_epoch * n_train_batches * patience[1])

            # if so, replace the old one, and compute the score
            # on the test dataset
            validation_scores[0] = this_validation_loss
            validation_scores[1] = numpy.mean(test_losses)
            validation_scores[2] = current_epoch
            validation_scores[3] = theta_value

        # We have exhausted the available patience,
        # so we need to stop training
        if patience <= current_epoch * n_train_batches:
            return False
        else:
            return True

    ###############
    # TRAIN MODEL #
    ###############

    print '... training the model'
    start_time = time.clock()

    # Select optimization method
    opt = None
    if optimization == 'nlcg':
        opt = climin.NonlinearConjugateGradient(numpy.zeros((n_in + 1) * n_out, dtype=x.dtype), train_fn, train_fn_grad)

    elif optimization == 'lbfgs':
        opt = climin.Lbfgs(numpy.zeros((n_in + 1) * n_out, dtype=x.dtype), train_fn, train_fn_grad)

    for info in opt:
        if (not callback(opt.wrt, info['n_iter'])) or (info['n_iter'] >= n_epochs - 1):
            break

    end_time = time.clock()

    print(
        (
            'Optimization complete with best validation score of %f %%, with '
            'test performance %f %%'
        )
        % (validation_scores[0] * 100., validation_scores[1] * 100.)
    )

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    best_W = validation_scores[3][0:n_in * n_out].reshape((n_in, n_out))
    image = Image.fromarray(
    tile_raster_images(best_W.T,
                           img_shape=(28, 28), tile_shape=(2, 5),
                           tile_spacing=(1, 1)))
    image.save('repflds.png')

    # Plot the errors against the epochs
    epochs = numpy.arange(0, n_epochs)
    plt.plot(epochs, trainings, 'b', epochs, validations, 'g', epochs, tests, 'r')
    green_circle, = plt.plot(validation_scores[2], validation_scores[0] * 100., 'o', mec='g', ms=15, mew=1, mfc='none',
                             label="Best Validation Error")
    # Create plot legend
    blue_patch = mpatches.Patch(color='blue', label='Train')
    green_patch = mpatches.Patch(color='green', label='Validation')
    red_patch = mpatches.Patch(color='red', label='Test')
    plt.legend(handles=[blue_patch, green_patch, red_patch, green_circle], numpoints = 1)
    plt.savefig('error.png')

if __name__ == '__main__':
    custom_optimization_mnist()