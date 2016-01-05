import matplotlib.pyplot as plt
from itertools import product
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

from pca_theano import PCA
import theano
import theano.tensor as T

mnist = fetch_mldata("MNIST original")
X_train, y_train = mnist.data[:60000] / 255., mnist.target[:60000]

X_train, y_train = shuffle(X_train, y_train)
# X_train, y_train = X_train[:5000], y_train[:5000]  # lets subsample a bit for a first impression

pca = PCA(M=2)

X = T.matrix('X', dtype='float64')

# Theano function which fits the model to the
# data i.e. applies dimensionality reduction
transform = theano.function(
    inputs=[X],
    outputs=pca.transform(X),
)

fig, plots = plt.subplots(10, 10)
fig.set_size_inches(50, 50)
plt.prism()
for i, j in product(xrange(10), repeat=2):
    if i > j:
        continue

    X_ = X_train[(y_train == i) + (y_train == j)]
    y_ = y_train[(y_train == i) + (y_train == j)]

    X_transformed = transform(X_)

    plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
    plots[i, j].set_xticks(())
    plots[i, j].set_yticks(())

    plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
    plots[j, i].set_xticks(())
    plots[j, i].set_yticks(())
    if i == 0:
        plots[i, j].set_title(j)
        plots[j, i].set_ylabel(j)

    #plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
plt.tight_layout()
plt.savefig("scatterplotMNIST.png")