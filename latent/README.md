PCA:
- pca_theano.py: implements the PCA class
- PCA_MNIST_Plot.py: Plots the MNIST dataset using the PCA class from pca_theano
- PCA_CIFAR_Plot.py: Plots the CIFAR dataset using the PCA class from pca_theano

Autoencoders:
- autoencoder.py: Trains a `non-sparse` autoencoder on the MNIST dataset
- autoencoder_sparse.py: Trains a sparse autoencoder on MNIST dataset. However, due to computing power limitations, the sparse auto encoder is trained only on the first 10000 training samples, which (hypothetically speaking) leads to the noisy receptive fields produced by the sparse autoencoder, although the reconstruction error achieved using the sparse autoencoder is less than the non-sparse autoencoder by at least a magnitude of 10