import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from plots import *
import sys
import load_data
import os
import time
import pandas as pd

config_file = sys.argv[1]
params = load_data.read_config(config_file)

print(params)

plot_folder = params['out_folder']
os.makedirs(''.join([plot_folder,'test_sigma/']),exist_ok=True)

## Load in results.mat
results_path = ''.join([plot_folder,'/results.mat'])
results = hdf5storage.loadmat(results_path, variable_names=['template', 'template_embedding'])
template = results['template']
template_embedding = results['template_embedding']

# import h5py
# template_idx = np.array(h5py.File(results_path)['template_idx'])
# import pdb; pdb.set_trace()

k=10
print("Embedding k fold template")
print(template.shape)
print(template_embedding.shape)
template_shuffled = np.random.permutation(np.append(template_embedding, template, axis=1))
split_size = np.floor(template_shuffled.shape[0]/k)
k_pred_embedding = np.empty((0,2))
for i in range(k):
    train = np.delete(template_shuffled, range(int(i*split_size), int((i+1)*split_size)), axis=0)
    test = template_shuffled[int(i*split_size) : int((i+1)*split_size), :]
    from KNNEmbed import KNNEmbed
    start = time.time()
    reembedder = KNNEmbed(k=5)
    reembedder.fit(train[:,2:],train[:,0:2])
    k_pred_embedding = np.append(k_pred_embedding,
                                 reembedder.predict(test[:,2:],weights='distance'),
                                 axis=0)
    print("Total Time ReEmbedding: ", time.time()-start)

f = plt.figure()
plt.scatter(template_embedding[:,0], template_embedding[:,1], marker='.', s=3, linewidths=0,
                c='b')
plt.scatter(k_pred_embedding[:,0], k_pred_embedding[:,1], marker='.', s=3, linewidths=0,
                c='m')
plt.legend()
plt.savefig(''.join([plot_folder,'k_fold_mbed','.png']), dpi=400)
plt.close()


[features, batch_ID] = load_data.load_data(params['analysis_path'],params['preds_path'],params['batch_name'],subsample=30)
# template_idx = hdf5storage.loadmat(results_path, variable_names=['template_idx'])['template_idx']
print("Testing kernal sigma for final embedding")
from KNNEmbed import KNNEmbed
start = time.time()
reembedder = KNNEmbed(k=5)
reembedder.fit(template,template_embedding)
final_embedding = reembedder.predict(features,weights="distance")
print("Total Time ReEmbedding: ", time.time()-start)

sigmas = [15, 25, 35, 45, 50]
for sigma in sigmas:
    filename = ''.join([plot_folder, 'test_sigma/final_embedding_sigma_',str(sigma)])
    # embed_scatter(final_embedding, filename=filename)
    _, _, _, density_map = clustering(final_embedding, filename, sigma=sigma, bins_per_edge=5000)

for batch in np.unique(batch_ID):
        # embed_scatter(final_embedding, filename =''.join([plot_folder,'byID/','final_',str(batch)]), colorby=np.where(batch_ID==batch,1,0))
        f = plt.figure()
        plt.scatter(final_embedding[:,0], final_embedding[:,1], marker='.', s=3, linewidths=0,
                    c='y', alpha=0.4)
        plt.scatter(final_embedding[batch_ID==batch,0], final_embedding[batch_ID==batch,1], 
                    marker='.', s=10, linewidths=0, c='k', alpha=1)
        plt.savefig(''.join([plot_folder,'byID/','final_', str(batch),'.png']), dpi=400)
        plt.close()
        # embed_scatter(template_embedding, filename =''.join([plot_folder,'byID/','temp_',str(batch)]), colorby=np.where(batch_ID[template_idx]==batch,1,0))