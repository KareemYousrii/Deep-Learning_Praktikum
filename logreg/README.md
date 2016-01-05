Multiclass logistic regression

VanillaLogReg.py: Implements logistic regression using stochastic gradient descent
LogRegOpt.py: Implements logistic regression using either NonLinearConjugateGradient or LBFGS from climin

Problem 12: Early stopping was used to terminate training using the notion of patience i.e. the number of iterations we will persist even though we are not achieving any improvement in terms of validation error. When the patience is depleted, the training simply stops. This patience is updated along the training, in a sense that the better validation error the model was capable of achieving is rewarded by a higher confidence in the model’s ability to achieve an even better validation error.

Bonus Question: The very last problem is very bad scientific practice since by tuning
your model to achieve a certain test error, information about the test dataset leaks
into the model, which then renders the performance of the dataset on the model as a
useless indicator as to how good your model really is.