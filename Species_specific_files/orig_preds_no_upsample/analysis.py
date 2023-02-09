import tsnecuda as tc
# import h5py
import numpy as np
import hdf5storage
import time
import sys
import math 
from plots import *

embedding_method = sys.argv[1]
print(embedding_method)

plot_folder = './initial_tadross_analysis/'

## Load in data (we only care about animalID, frames_with_good_tracking and jt_features)
# analysis_file_path = './myanalysisstruct.mat'
# predictions_file_path = './predictions.mat'
# id_name = 'animalID'

analysis_file_path = '../../CAPTURE_data/full_tadross_data/anstruct_notsne_fromlocal.mat'
predictions_file_path = '../../CAPTURE_data/full_tadross_data/merged_predictions.mat'
id_name = 'recordingID'

analysisstruct = hdf5storage.loadmat(analysis_file_path, variable_names=['jt_features','frames_with_good_tracking'])
animal_ID = np.squeeze(hdf5storage.loadmat(predictions_file_path, variable_names=[id_name])[id_name].astype(int))
features_full = analysisstruct['jt_features']
frames_with_good_tracking = np.squeeze(analysisstruct['frames_with_good_tracking'][0][0].astype(int))

animal_IDs_good_tracking = animal_ID[frames_with_good_tracking] # Indexing out animal IDs

# Subsample by 30
features = features_full[::30]
animal_IDs_good_tracking = animal_IDs_good_tracking[::30]

## Looping through the condensed feature set and embedding in batches
start = time.time()
template = np.empty((0,np.shape(features)[1]))
template_idx = []
for animal in np.unique(animal_IDs_good_tracking):
    # import pdb; pdb.set_trace()
    features_ID = features[np.where(animal_IDs_good_tracking == animal)[0],:]
    n = np.shape(features_ID)[0]

    if embedding_method == 'tsne':
        print("Running sklearn tSNE on each animal video")

        from sklearn.manifold import TSNE
        tsne = TSNE(n_iter=1000, perplexity=max(int(n/100),30), init='random', n_jobs=12, verbose=3, method='barnes_hut')
        embedding = tsne.fit_transform(features_ID)
        
        filename = ''.join([plot_folder, 'tsne_byID_', str(animal)])
        embed_scatter(embedding, filename = filename)

    elif embedding_method == 'tsne_split':
        print("Running sklearn tSNE on each animal video split into batches of ~50k frames")
        split_factor = math.floor(n/50000)

        for i in range(split_factor):
            if i == split_factor-1:
                split_features = features_ID[i*math.floor(n/split_factor):end,:]
                print(np.shape(split_features))
            else:
                split_features = features_ID[i*math.floor(n/split_factor):(i+1)*math.floor(n/split_factor),:]
                print(np.shape(split_features))

            from sklearn.manifold import TSNE
            tsne = TSNE(n_iter=1000, perplexity=max(int(n/100),30), init='random', n_jobs=12, verbose=3, method='barnes_hut')
            embedding = tsne.fit_transform(split_features)

            filename = ''.join([plot_folder,'tsne_by50k_',str(animal),'_',str(split_factor)])
            embed_scatter(embedding, filename=filename)

    elif embedding_method == 'tsne_cuda':
        print("Running tSNE cuda on each animal video ", str(animal))

        import tsnecuda as tc
        tsne = tc.TSNE(n_iter=1000, verbose=2, num_neighbors=300, perplexity=int(n/100), learning_rate=int(n/12))
        embedding = tsne.fit_transform(features_ID)

        filename = ''.join([plot_folder, 'tsne_cuda_byID_', str(animal)])
        embed_scatter(embedding, filename=filename)

    elif embedding_method == 'umap':
        print("Running umap on each animal video")

        import umap
        umap_transform = umap.UMAP(n_neighbors=100, verbose=True)
        embedding = umap_transform.fit_transform(features_ID)

        filename = ''.join([plot_folder, 'umap_byID_', str(animal)])
        embed_scatter(embedding, filename=filename)

    watershed_map, data_by_cluster = clustering(embedding, filename=filename)
    sampled_points, idx = sample_clusters(features_ID, data_by_cluster, size=30)
    template = np.append(template, sampled_points, axis=0)
    template_idx += idx

print("Total Time: ", time.time()-start)
print(type(template_idx[0]))
reembedding = reembed(template, template_idx, features_full, method=embedding_method, plot_folder=plot_folder)



# predictions = read_mat(predictions_file_path)

# f = h5py.File(analysis_file_path)
# print(f.items())
# jt_features = np.array(f['jt_features'])
# print(np.shape(jt_features))



# num_neighbors = [50,100]
# perplexity = [70,400000]

# for nn in num_neighbors:
#     for p in perplexity:
#         try:
#             tsne = tc.TSNE(n_iter=1000, verbose=2, num_neighbors=nn,perplexity=p)
#             tsne_results = tsne.fit_transform(jt_features.T)
#             print("worked on " + str(nn) + " " + str(p))
#         except:
#             print("failed on " + str(nn) + " " + str(p))
#             continue

## Clustering
#pip install mpl-scatter-density
# Could also just do np.2dhistogram with high bins and interpolate for watershed clustering