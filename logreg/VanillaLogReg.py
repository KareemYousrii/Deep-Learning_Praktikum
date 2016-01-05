import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


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

        # Initialize the weights as a matrix W,
        # where each column of W holds a training
        # sample
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        # Initialize the bias vector b
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # This is analogous to computing P(C_k|\phi): THe posterior probability
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # Choose the class with the highest probability as the predicted class
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
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

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

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


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    :type learning rate: float
    :param learning_rate: learning rate used

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the pickled MNIST dataset file
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Compute the number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

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
    cost = classifier.negative_log_likelihood(y)

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
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Compute the gradient of cost w.r.t the weights and biases
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # Specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # Compiling a Theano function that returns the cost, as well as
    # updates the parameters of the models based on the rules defined
    # in 'updates' i.e. performs the actual training of the model
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############

    print '... training the model'

    # early-stopping parameters
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    # Go through this many minibachtches before checking the validation error.
    # In this case, we validate our training every after every training epoch.
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf  # Initially, we set our validation error to inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    # We keep looping as long as we haven't exceeded the maximum
    # number of training epochs, and we haven't run out of patience
    # i.e. the number of iterations in which the validation error
    # hasn't improved < patience
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        # We loop over our training batches
        for minibatch_index in xrange(n_train_batches):

            # The cross-entropy error returned by the training
            # function
            minibatch_avg_cost = train_model(minibatch_index)

            # An unnecessarily smart way of keeping track of the
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            # After every epoch, validate our model, obtaining
            # the zero-loss error.
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100
                    )
                )

                # update best validation score
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            'epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()

    print(
        (
            'optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )

    print 'The code was run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


if __name__ == '__main__':
    sgd_optimization_mnist()