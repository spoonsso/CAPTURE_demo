import tsnecuda as tc
# import h5py
import numpy as np
import time
import sys
import math 
from plots import *
import load_data
import os

# os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/exx/miniconda3/envs/capture/lib/')

config_file = sys.argv[1]
params = load_data.read_config(config_file)

print(params)

plot_folder = params['out_folder']
os.makedirs(''.join([plot_folder,'byID/']),exist_ok=True)
[features, batch_ID] = load_data.load_data(params['analysis_path'],params['preds_path'],params['batch_name'],subsample=20)

## Looping through the condensed feature set and embedding in batches
for embedding_method in params['embedding_method']:
    start = time.time()
    template = np.empty((0,np.shape(features)[1]))
    template_idx = []
    filename = ''.join([plot_folder,'/batch_maps/', embedding_method, '_byID_'])
    for batch in np.unique(batch_ID):
        features_ID = features[batch_ID == batch,:]

        print("Size of video")
        print(np.shape(features_ID))

        n = np.shape(features_ID)[0]

        if embedding_method == 'tsne':
            print("Running sklearn tSNE on each batch")

            from sklearn.manifold import TSNE
            tsne = TSNE(n_iter=1000, perplexity=max(int(n/100),30), init='random', n_jobs=12, verbose=3, method='barnes_hut')
            embedding = tsne.fit_transform(features_ID)
            embed_scatter(embedding, filename = ''.join([filename, str(batch)]))

        elif embedding_method == 'tsne_split':
            print("Running sklearn tSNE on each batch video split into sub-batches of ~50k frames")
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
                embed_scatter(embedding, filename=''.join([filename,str(batch),'_',str(split_factor)]))

        elif embedding_method == 'tsne_cuda':
            print("Running tSNE cuda on each batch video")

            import tsnecuda as tc
            tsne = tc.TSNE(n_iter=1500, verbose=2, num_neighbors=300, perplexity=int(n/100), learning_rate=int(n/12))
            embedding = tsne.fit_transform(features_ID)
            embed_scatter(embedding, filename=''.join([filename, str(batch)]))

        elif embedding_method == 'umap':
            print("Running umap on each batch video")

            import umap
            umap_transform = umap.UMAP(n_neighbors=100, verbose=True)
            embedding = umap_transform.fit_transform(features_ID)
            embed_scatter(embedding, filename=''.join([filename, str(batch)]))

        watershed_map, _, data_by_cluster, _ = clustering(embedding, filename=''.join([filename, str(batch)]), sigma=15)
        sampled_points, idx = sample_clusters(features_ID, data_by_cluster, size=20)
        idx = np.nonzero(batch_ID==batch)[0][idx]
        template = np.append(template, sampled_points, axis=0)
        template_idx += list(idx)

    print("Total Time: ", time.time()-start)
    final_embedding, temp_embedding = reembed(template, template_idx, features, method=embedding_method, plot_folder=plot_folder)

    # for batch in np.unique(batch_ID):
    #     embed_scatter(final_embedding, filename =''.join([plot_folder,'byID/','final_',str(batch)]), colorby=np.where(batch_ID==batch,1,0))
    #     embed_scatter(temp_embedding, filename =''.join([plot_folder,'byID/','temp_',str(batch)]), colorby=np.where(batch_ID[template_idx]==batch,1,0))

    # import pdb; pdb.set_trace()





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