Neural Networks:

nn.py: implements vanilla neural network with stochastic gradient descent. L1-regularization, weight decay, dropout as well as momentum were also implemented. Three types of activation functions can be used: the logistic sigmoid, tanh as well as the relu.

nn_opt.py: implements vanilla neural network which is usable with other optimisation methods provided from the climin package. However, this implementation is currently not working properly.

Using `relu` I was able to achieve validation and test score which are much lower than those I was able to achieve using `tanh` or `logistic sigmoid` activation function. I managed to achieve a validation error of 1.85 and a test error of 1.97 in only 250 epochs.