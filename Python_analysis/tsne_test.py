import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage import measure
import numpy as np
import hdf5storage
from plots import *
results_path = './initial_tadross_analysis/results.mat'
plot_folder = './initial_tadross_analysis/'

results = hdf5storage.loadmat(results_path, 
                              variable_names=['template'])
template = results['template']

# import tsnecuda as tc
# n = np.shape(template)[0]
# tsne = tc.TSNE(n_iter=5000, verbose=2, num_neighbors=200, perplexity=int(n/100), learning_rate=int(n/12))
# temp_embedding = tsne.fit_transform(template)
# filename = ''.join([plot_folder, 'tsne_cuda_template'])
# embed_scatter(temp_embedding, filename=filename)
# clustering(temp_embedding, filename)

import umap
umap_transform = umap.UMAP(n_neighbors=300, verbose=True)
temp_embedding = umap_transform.fit_transform(template)
filename = ''.join([plot_folder, 'umap_template'])
embed_scatter(temp_embedding, filename=filename)