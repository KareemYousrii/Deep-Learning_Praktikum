import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import climin

from logistic_sgd import LogisticRegression, load_data

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W, b):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1

        self.W = W
        self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, theano_rng, input, n_in, n_out, is_train, W=None, b=None, activation=T.tanh, p=1.):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation functions. The weight matrix W is of shape
        (n_in, n_out) and the bias vector v is of shape (n_out,).

        The hidden unit activation is given by: tanh(dot(input, W) + b)

        :param rng: a random number generator used to initialize the weights.
        If the features were initialized to the same values, they would have
        the same gradients, and would end up learning the same non-linear
        transformation of the inputs.

        :param input: a symbolic tensor of shape (n_examples, n_in)
        :param n_in: dimensionality of the input
        :param n_out: number of hidden units
        :param W: A matrix of weights connecting the input units to
        the hidden units. It has a shape of (n_in, n_out)
        :param b: A vector of biases for the hidden units. It is of
        shape (n_out,)
        :param activation: The activation function to be applied on the
        hidden units.
        :param p: The probability that a hidden unit is retained i.e. the
         probability of dropping out a hidden unit is given by (1 - p)
        """
        self.input = input

        # `W` is initialized with values uniformly sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for the tanh activation function. The reason for such an
        # initialization is to prevent the neurons from saturating
        # e.g. The logistic function is approximately flat for large
        # positive and large negative inputs. The derivative of the
        # logistic function at 2 is almost 1/10, but at 10, the derivative
        # is almost 1/22000 i.e. a neuron with an input of 10 will learn
        # 2200 times(!!!) slower than a neuron with an input of 2. Using
        # the sampling above proposed by Bengio et. al, we circumvent this
        # problem by limiting the weights to always lie in a "small enough"
        # range for the hidden units not to saturate.

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        # Applying the non-linearity to the Weighted output
        out = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # Apply dropout to the output of the hidden layer
        mask = theano_rng.binomial(n=1, p=p, size=out.shape, dtype=theano.config.floatX)

        # If is_trained is set to 1, we are currently training the model, and therefore,
        # we should use dropout. Otherwise, we are testing and should therefore scale
        # the outputs. From the original paper on dropout:
        # If a unit is retained with probability p during training, the outgoing weights
        # of that unit are multiplied by p at test time. Finally cast the output to
        # float32
        self.output = ifelse(T.neq(is_train, 0), T.cast(mask * out,theano.config.floatX),
                             T.cast(p * out, theano.config.floatX))

        self.params = [self.W, self.b]

class MLP(object):
    """Multi-Layer Perceptron Class

        A multilayer perceptron is a feedforward artificial neural network
        model that has one layer or more of hidden units and nonlinear
        activations. In this case, the hidden layers are defined by a
        ``HiddenLayer`` class utilising either the tanh or logistic
        sigmoid function as the non-linearity whereas, the output layer
        is defined by the ``LogisticRegression`` class which utilizes the
        softmax function as its activation function
    """

    def __init__(self, rng, theano_rng, input, n_in, n_hidden, n_out, is_train):
        """
        :param rng: a random number generator used to initalize the weights

        :param input: a symbolic variable that describes the input of the
        architecture, in this case a minibatch

        :param n_in: dimensionality of the inputs

        :param n_hidden: number of hidden units

        :param n_out: number of output units
        """
        theta_value = numpy.concatenate(
            (
                numpy.asarray(
                    rng.uniform(
                        low=-numpy.sqrt(6./(n_in + n_hidden)),
                        high=numpy.sqrt(6./(n_in + n_hidden)),
                        size=(n_in*n_hidden)
                    )
                ),
                numpy.zeros(n_hidden + (n_hidden + 1) * n_out)
            )
        )

        # theta_value = numpy.zeros(((n_in + 1) * n_hidden) + ((n_hidden + 1) * n_out))

        self.theta = theano.shared(
            value=theta_value,
            name='theta',
            borrow=True
        )

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            theano_rng=theano_rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            is_train=is_train,
            W=self.theta[0:n_in*n_hidden].reshape((n_in, n_hidden)),
            b=self.theta[n_in*n_hidden: n_in * n_hidden + n_hidden],
            activation=T.tanh
        )

        base_index = n_in * n_hidden + n_hidden
        # The logistic regression layers gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            W=self.theta[base_index: base_index + (n_hidden * n_out)].reshape((n_hidden, n_out)),
            b=self.theta[base_index + (n_hidden * n_out):]
        )

        # L1 norm; one regularization option is to enforce the L1 norm
        # to be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # Square of L2 norm; one regularization option is to enforce the
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # The negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the logistic
        # regression layer.
        #
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        # Same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # The parameters of the model are compromised of the parameters
        # of the hidden layer as well as the logistic regression layer
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20,n_in = 28*28, n_hidden=500, n_out =10, momentum_coeff=0.,
             optimization='nlcg'):
    """

    :param learning_rate: learning rate used for the parameters
    :param L1_reg: lambda for the L1 regularization
    :param L2_reg: lambda for the L2-squared regularization
    :param n_epochs: number of epochs on which to train the data.
    :param dataset: pickled mnist data file
    :param batch_size: size of the mini-batch to be used with
    sgd
    :param n_hidden: number of hidden units
    :param momentum_coeff: Controls the amount of damping of the velocity
    as a result of previous gradients in sgd
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print train_set_x.shape
    print valid_set_x.shape
    print test_set_x.shape

    # Compute the number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Allocate symbolic variables for the data
    index = T.lscalar() # index to minibatch
    x = T.matrix('x')
    y = T.ivector('y')

    is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

    rng = numpy.random.RandomState(1234)
    theano_rng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out,
        is_train=is_train
    )

    # The cost that we minimize during training is the negative log likelihood
    # of the model plus the regularization terms
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # We compile a Theano function that computes the mistakes that are
    # made by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.asarray([1], dtype='int32')[0]
        },
        name="test"
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.asarray([1], dtype='int32')[0]
        },
    )

    training_loss = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index: index + batch_size],
            y: train_set_y[index: index + batch_size],
            is_train: numpy.asarray([1], dtype='int32')[0]
        }
    )

    # compile a theano function that returns the cost of a minibatch
    batch_cost = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index: index + batch_size],
            y: train_set_y[index: index + batch_size],
            is_train: numpy.asarray([1], dtype='int32')[0]
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
            y: train_set_y[index: index + batch_size],
            is_train: numpy.asarray([1], dtype='int32')[0]
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
    patience = [10000, 2]
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
        print this_validation_loss
        validations.append(this_validation_loss * 100.)

        # compute the test loss
        test_losses = [test_model(i*batch_size)
                       for i in xrange(n_test_batches)]
        this_test_loss = numpy.mean(test_losses)
        print this_test_loss
        tests.append(this_test_loss * 100.)

        # compute the training loss
        training_losses = [training_loss(i * batch_size)
                           for i in xrange(n_train_batches)]
        this_train_loss = numpy.mean(training_losses)
        print this_train_loss
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

    opt = climin.RmsProp(numpy.concatenate((numpy.asarray(rng.uniform(low=-numpy.sqrt(6./(n_in + n_hidden)), high=numpy.sqrt(6./(n_in + n_hidden)), size=(n_in*n_hidden))), numpy.zeros(n_hidden + (n_hidden + 1) * n_out))), train_fn_grad, 0.3)
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

if __name__ == '__main__':
    test_mlp()