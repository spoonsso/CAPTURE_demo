# from tsnecuda import TSNE
# import numpy as np

# a = np.random.rand(100,10)
# X = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(a)

from keras.datasets import mnist
import tsnecuda as tc
# import sklearn.manifold as skmn
# import matplotlib.pyplot as plt
import time
# import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train.shape)
print(x_train.shape)
print(x_train)
print(x_train.reshape(6000,-1).shape)

t = time.time()
tsne = tc.TSNE(n_iter=1000, verbose=2, num_neighbors=64)
tsne_results = tsne.fit_transform(x_train.reshape(60000,-1))
t2 = time.time()
print("Time for cuda tsne")
print(t2-t)

# X_embedded = skmn.TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(x_train.reshape(60000,-1))
# print("Time for sklearn tsne")
# print(time.time()-t2)

# print(tsne_results.shape)