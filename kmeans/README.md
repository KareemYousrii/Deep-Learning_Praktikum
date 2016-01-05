Options:
  -h, --help            show this help message and exit
  -k NUM_CLUSTERS, --num_clusters=NUM_CLUSTERS
                        The number of clusters to use in kmeans
  -e EPSILON_WHITENING, --epsilon_whitening=EPSILON_WHITENING
                        The epsilon to be used for ZCA whitening

Implementation for Kmeans and Kmeans mini-batch can be found in kmeans_Theano.py respectively. Furthermore, I utilise the file `utils.py` taken from the deep learning tutorial in order to visualise the receptive fields.

Regarding the repflds_mini-batch, it is my hypothesis that the gaps in the receptive fields are due to clusters which have collapsed since they contained no data points. This might be because `random` subsets of the dataset was used, and therefore, there may have been some data points to which the model was not exposed.

