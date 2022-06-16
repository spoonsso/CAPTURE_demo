import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from plots import *
import sys
import load_data
import os
import time
import pandas as pd
from dannce_vis_3d import *

config_file = sys.argv[1]
params = load_data.read_config(config_file)

print(params)

plot_folder = params['out_folder']

## Load in results.mat
results_path = ''.join([plot_folder,'/results.mat'])
results = hdf5storage.loadmat(results_path, variable_names=['template', 'template_embedding'])
template = results['template']
template_embedding = results['template_embedding']

subsample_rate = 10
[features, batch_ID] = load_data.load_data(params['analysis_path'],params['preds_path'],params['batch_name'],subsample=subsample_rate)

metadata = pd.read_csv(params['meta_path'])[['Condition']]
metadata = metadata.iloc[batch_ID].reset_index().rename(columns={'index':'batchID'})

filename = ''.join([plot_folder, 'tsne_cuda_template'])
embed_scatter(template_embedding, filename=filename)
clustering(template_embedding, filename, max_clip=1)

# template_idx = hdf5storage.loadmat(results_path, variable_names=['template_idx'])['template_idx']
print("Obtaining final embedding")
from KNNEmbed import KNNEmbed
start = time.time()
reembedder = KNNEmbed(k=5)
reembedder.fit(template,template_embedding)
final_embedding = reembedder.predict(features, weights='distance') #[np.where(batch_ID == 0 or batch_ID==1)[0],:]
print("Total Time ReEmbedding: ", time.time()-start)

bins_per_edge = 5000
sigma = 50
filename = ''.join([plot_folder, 'tsne_cuda_final'])
embed_scatter(final_embedding, filename=filename)
watershed_map, watershed_borders, data_by_cluster, density_map = clustering(final_embedding, filename, sigma=sigma, bins_per_edge=bins_per_edge, max_clip = 0.75)

print(np.amax(data_by_cluster))
print(np.unique(data_by_cluster).shape)
print(np.unique(data_by_cluster))

x_range = int(np.ceil(np.amax(final_embedding[:,0])) - np.floor(np.amin(final_embedding[:,0])))
y_range = int(np.ceil(np.amax(final_embedding[:,1])) - np.floor(np.amin(final_embedding[:,1])))
ax_range = [[int(np.floor(np.amin(final_embedding[:,0]))-x_range/40),int(np.ceil(np.amax(final_embedding[:,0]))+x_range/40)],
            [int(np.floor(np.amin(final_embedding[:,1]))-y_range/40),int(np.ceil(np.amax(final_embedding[:,1]))+y_range/40)]]

# import pdb; pdb.set_trace()
adj_embed = (final_embedding - np.array(ax_range)[:,0])*bins_per_edge/np.array([x_range+x_range/20,y_range+y_range/20])

f = plt.figure()
ax = f.add_subplot(111)
# ax.plot(watershed_borders[:,0],bins_per_edge-watershed_borders[:,1]-1,'.k',markersize=1)
ax.imshow(watershed_map, zorder=1, extent=[*ax_range[0], *ax_range[1]])
ax.plot(final_embedding[:,0], final_embedding[:,1],'.r',markersize=1,alpha=0.1, zorder=2)
# ax.imshow(density_by_c,cmap='hot_r')
ax.set_aspect('auto')
filename = ''.join([plot_folder,'points_by_cluster/all.png'])
plt.savefig(filename,dpi=400)
plt.close()

# ### CHECKING CLUSTER INDEXING
# for cluster in range(np.amax(data_by_cluster)+1):
#     print("Cluster_" + str(cluster))
#     embedding_by_cluster = final_embedding[data_by_cluster==cluster,:]
#     # density_by_c, _, _ = map_density(embedding_by_cluster,bins_per_edge=5000,sigma=10, max_clip=1)
#     f = plt.figure()
#     ax = f.add_subplot(111)
#     # ax.plot(watershed_borders[:,0],bins_per_edge-watershed_borders[:,1]-1,'.k',markersize=1)
#     ax.imshow(watershed_map,extent=[*ax_range[0], *ax_range[1]],zorder=0)
#     print(embedding_by_cluster.shape)
#     # import pdb; pdb.set_trace()

#     ax.plot(embedding_by_cluster[:,0], embedding_by_cluster[:,1],'.r',markersize=1,alpha=0.1, zorder=1)
#     # ax.imshow(density_by_c,cmap='hot_r')
#     ax.set_aspect('auto')
#     filename = ''.join([plot_folder,'points_by_cluster/points_',str(cluster),'.png'])
#     plt.savefig(filename,dpi=400)
#     plt.close()

batch_ID = metadata['Condition'].values.tolist()
os.makedirs(''.join([plot_folder,'density_by_condition/']),exist_ok=True)
for batch in set(batch_ID):
    embedding_by_ID = final_embedding[metadata.index[metadata['Condition']==batch].to_list(),:]
    density_by_ID, _, _ = map_density(embedding_by_ID,bins_per_edge=bins_per_edge,sigma=sigma, max_clip=0.3, x_range=x_range,y_range=y_range, hist_range=ax_range)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(watershed_borders[:,0],watershed_borders[:,1],'.k',markersize=0.05)
    ax.imshow(density_by_ID,cmap='hot_r')
    ax.set_aspect('auto')
    filename = ''.join([plot_folder,'density_by_condition/density_vid_',str(batch),'.png'])
    plt.savefig(filename,dpi=400)
    plt.close()


# batch_ID x cluster_ID
num_batches = len(set(batch_ID))
cluster_freqs = np.zeros((num_batches, np.max(data_by_cluster)+1))
for i, batch in enumerate(set(batch_ID)):
    cluster_by_ID = data_by_cluster[metadata.index[metadata['Condition']==batch].to_list()]
    cluster_freqs[i,:] = np.histogram(cluster_by_ID, bins=np.max(data_by_cluster)+1, range=(-0.5,0.5+np.max(data_by_cluster)))[0]

frame_totals = np.sum(cluster_freqs,axis=1)
max_frames = np.amax(frame_totals)

cluster_freqs = cluster_freqs*np.expand_dims(max_frames/frame_totals,axis=1) #Adjust so each batch has same total counts
cluster_freqs -= np.mean(cluster_freqs, axis=0)
cluster_freqs = cluster_freqs/np.clip(np.std(cluster_freqs,axis=0),1e-6, None)
sorted_indices = np.argsort(cluster_freqs, axis=1)

colors = ['tab:blue','tab:red','tab:purple','tab:brown','tab:green','tab:olive','tab:cyan','tab:pink','tab:orange','tab:gray']
for i, batch_i in enumerate(set(batch_ID)):
    f,ax_array = plt.subplots(2,1,figsize=(20,15))
    legend = np.empty((2,5))
    f.suptitle(''.join(['Highest and Lowest Differential Expression for ', batch_i]))
    for j, batch_j in enumerate(set(batch_ID)):
        ax_array[0].bar(np.arange(5)*(num_batches+1)+j, cluster_freqs[j,sorted_indices[i,-5:]], color=colors[j], label=str(batch_j))
        ax_array[1].bar(np.arange(5)*(num_batches+1)+j, cluster_freqs[j,sorted_indices[i,:5]], color=colors[j], label=str(batch_j))

    ax_array[0].set_xticks(np.arange(5)*(num_batches+1)+3, sorted_indices[i,-5:])
    ax_array[0].legend()
    ax_array[0].set_xlabel('Behavioral Cluster')
    ax_array[0].set_ylabel('Behavioral Expression')
    ax_array[0].set_title('Upregulated')

    ax_array[1].set_xticks(np.arange(5)*(num_batches+1)+3, sorted_indices[i,:5])
    ax_array[1].legend()
    ax_array[1].set_xlabel('Behavioral Cluster')
    ax_array[1].set_ylabel('Behavioral Expression')
    ax_array[1].set_title('Downregulated')
    plt.savefig(''.join([plot_folder, 'diff_exp_condition/', str(batch_i),'.png']),dpi=400)
    plt.close()


_, template_idx = sample_clusters(features,data_by_cluster,size=10)
template_idx = np.reshape(np.array(template_idx)*subsample_rate, (int(len(template_idx)/10),10))


preds = load_predictions(PRED_EXP = params['preds_path'])
for i,row in enumerate(template_idx):
    skeleton_vid3D(preds,
                   frames = list(row),
                   VID_NAME = ''.join(['cluster_',str(np.unique(data_by_cluster)[i]),'.mp4']),
                   EXP_ROOT = ''.join([plot_folder, 'skeleton_vids/']))