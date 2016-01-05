import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.ifelse import ifelse

from logistic_sgd import LogisticRegression, load_data
from utils import tile_raster_images

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

try:
    import PIL.Image as Image
except ImportError:
    import Image

class HiddenLayer(object):
    def __init__(self, rng, theano_rng, input, n_in, n_out, is_train, W=None, b=None, activation='tanh', p=0.5):
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

        if activation == 'tanh':
            activation = T.tanh
        elif activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif activation == 'relu':
            activation = lambda x: x * (x > 0)
        else:
            raise NotImplementedError

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
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6./(n_in + n_out)),
                    high=numpy.sqrt(6./(n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)


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

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            theano_rng=theano_rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            is_train=is_train,
            activation='tanh'
        )

        # The logistic regression layers gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
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
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=300, momentum_coeff=0.):
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

    # Compute the number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0] / batch_size

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
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
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
            is_train: numpy.asarray([0], dtype='int32')[0]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.asarray([0], dtype='int32')[0]
        }
    )

    train_loss = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: numpy.asarray([0], dtype='int32')[0]
        }
    )

    # Compute the gradient of the cost w.r.t theta
    #check
    gparams = [T.grad(cost, param) for param in classifier.params]

    # # specify how to update the parameters of the model as a list of
    # # (variable, update expression) pairs
    # updates = [
    #     (param, param - learning_rate * gparam)
    #     for param, gparam in zip(classifier.params, gparams)
    # ]

    # List of updates for every set of parameters
    updates = []

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    for param, gparam in zip(classifier.params, gparams):

        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" previous gradients i.e. when the previous momenta
        # have the same direction, this contributes to the velocity of the gradient descent
        # and therefore, we take larger steps. Here, the velocity `dict` tracks old gradients.

        velocity = theano.shared(theano._asarray(param.get_value()*0., dtype=theano.config.floatX))
        updated_velocity = momentum_coeff * velocity - learning_rate * gparam

        updates.append((velocity, updated_velocity))
        updates.append((param, param + updated_velocity))


    # compiling a Theano function which returns the cost, but at the
    # same time updates the parameters of the model
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size],
            is_train: numpy.asarray([1], dtype='int32')[0]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############

    print '... training'

    # early-stopping parameters
    patience = 10000  # The number of iterations to execute regardless of the validation error
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    best_W = None
    best_epoch = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False

    # Keeping track of training, testing and validation errors
    # per epoch
    validations = []
    tests = []
    trainings = []

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            # A fancy way of keeping track of the current iteration
            iter = (epoch - 1) * n_train_batches + minibatch_index

            # Check the validation error every validation frequency
            # (in this case, we check every epoch)
            if (iter + 1) % validation_frequency == 0:

                # Compute the validation error i.e. the zero-one
                # loss on the validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]

                # The validation error is the mean over all the minibatches
                # of the validation set
                this_validation_loss = numpy.mean(validation_losses)

                # test the current model using the test set,
                # averaging over the test scores obtained by
                # all minibatches
                test_losses = [test_model(i) for i
                               in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)

                # The error achieved by the current model
                # on the training dataset
                train_losses = [train_loss(i) for i
                                in xrange(n_train_batches)]
                train_score = numpy.mean(train_losses)

                # For plotting error curve
                validations.append(this_validation_loss * 100)
                tests.append(test_score * 100)
                trainings.append(train_score * 100)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # Maintain global best for validation loss
                if this_validation_loss < best_validation_loss:

                    # If the improvement in the validation loss surpasses
                    # the improvement threshold, we allow an increase in
                    # patience
                    if(this_validation_loss <
                               best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # Update the global best
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    best_W = classifier.hiddenLayer.W.get_value(borrow=False)
                    best_epoch = epoch

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break


    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    image = Image.fromarray(
    tile_raster_images(best_W.T,
                           img_shape=(28, 28), tile_shape=(3, 10),
                           tile_spacing=(1, 1)))
    image.save('repflds.png')

    # Plot the errors against the epochs
    epochs = numpy.arange(1, n_epochs + 1)
    plt.plot(epochs, trainings, 'b', epochs, validations, 'g', epochs, tests, 'r')
    green_circle, = plt.plot(best_epoch, best_validation_loss * 100., 'o', mec='g', ms=15, mew=1, mfc='none',
                             label="Best Validation Error")

    # Create plot legend
    blue_patch = mpatches.Patch(color='blue', label='Train')
    green_patch = mpatches.Patch(color='green', label='Validation')
    red_patch = mpatches.Patch(color='red', label='Test')
    plt.legend(handles=[blue_patch, green_patch, red_patch, green_circle], numpoints = 1)
    plt.savefig('error.png')


if __name__ == '__main__':
    test_mlp()