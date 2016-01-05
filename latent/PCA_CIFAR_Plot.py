import matplotlib.pyplot as plt
from itertools import product
from sklearn.utils import shuffle

from pca_theano import PCA
import theano
import theano.tensor as T

import cPickle
import numpy as np

def rgb2gray(imgs):
    gray_imgs = np.empty((len(imgs), 1024))

    for i in xrange(len(imgs)):
        rgb = imgs[i, :].reshape(3, 32, 32)
        r, g, b = rgb[:3, ...]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray_imgs[i, :] = gray.flatten()

    return gray_imgs

d = open('data_batch_1', 'rb')
l = open('batches.meta')
dict = cPickle.load(d)
label_names = cPickle.load(l)
d.close()
l.close()

X_train, y_train = dict['data'] / 255., np.asarray(dict['labels'])
label_names = label_names['label_names']

X_train, y_train = shuffle(X_train, y_train)

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
        plots[i, j].set_title(label_names[j])
        plots[j, i].set_ylabel(label_names[j])

plt.tight_layout()
plt.savefig("scatterplotCIFAR.png")