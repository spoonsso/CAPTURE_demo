import tsnecuda as tc
import numpy as np
import time
import sys
import math 
from plots import *
import load_data
import os
import pandas as pd
import hdf5storage
from KNNEmbed import KNNEmbed
from validation import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import kl_div
from scipy.spatial import distance



config_file = sys.argv[1]
params = load_data.read_config(config_file)

print(params)

plot_folder = ''.join([params['out_folder'],'tsne_param_figures/sigma/'])

###### LOAD TEMPLATE EMBEDDING DATA ######
results_path = ''.join([params['out_folder'],'/results.mat'])
results = hdf5storage.loadmat(results_path, variable_names=['template'])#, 'template_embedding'])
template = results['template']

###### LOAD FULL DATA ######
subsample_rate = 30
[features, batch_ID] = load_data.load_data(params['analysis_path'],params['preds_path'],params['batch_name'],subsample=subsample_rate)

###### LOAD METADATA FOR DATA ######
meta_vid = pd.read_csv(params['meta_path'])[['Condition','AnimalID','Timepoint']]
metadata = meta_vid.iloc[batch_ID].reset_index().rename(columns={'index':'batchID'})
condition_names = ['Baseline','Lesion2']

features = features[metadata.index[(metadata['Condition'].isin(condition_names)) & 
                                   (metadata['Timepoint']==1.25)].to_list(),:]
batch_ID = batch_ID[metadata.index[((metadata['Condition'].isin(condition_names)) & 
                                   (metadata['Timepoint']==1.25))].to_list()]

metadata = metadata.loc[(metadata['Condition'].isin(condition_names)) &
                         (metadata['Timepoint']==1.25)].reset_index()

meta_vid = meta_vid.loc[(meta_vid['Condition'].isin(condition_names)) &
                        (meta_vid['Timepoint']==1.25)].reset_index().rename(columns={'index':'vidID'})

# import pdb; pdb.set_trace()

bins_per_edge = 1000
best_sigma = 15
n = np.shape(template)[0]

sigma_range = range(3,21,1)
tnn_range = [10,30,50,100,200,300,500,1000]
pp_range = [10,20,30,70,100,250,500,int(n/100),1000]
knn_range = [1,3,5,7,9,11,13,15,17,19,21]

param_range = sigma_range

all_cos_mean,all_cos_se = [],[]
pd_cos_mean, pd_cos_se = [],[]
base_cos_mean, base_cos_se = [],[]
neg_cos_mean, neg_cos_se = [],[]
n_clusters = []
# for i in range(len(param_range)):
###### COMPLEXITY OF TEMPLATE EMBEDDING ######
import tsnecuda as tc
# tsne = tc.TSNE(n_iter=2500, verbose=2, num_neighbors=tnn_range[i], perplexity=int(n/100))#, learning_rate=int(n/12)) #TNN
tsne = tc.TSNE(n_iter=2500, verbose=2, num_neighbors=200, perplexity=70)#, learning_rate=int(n/12)) #PP
template_embedding = tsne.fit_transform(template)

# for i in range(len(param_range)):
min_err_nn=5 
print("Obtaining final embedding")
from KNNEmbed import KNNEmbed
start = time.time()
reembedder = KNNEmbed(k=min_err_nn)
reembedder.fit(template,template_embedding)
final_embedding = reembedder.predict(features, weights='distance')
print("Total Time ReEmbedding: ", time.time()-start)

for i in range(len(param_range)):
    watershed_map, watershed_borders, data_by_cluster, density_map = clustering(final_embedding, 
                                                                                filename=None, 
                                                                                sigma=param_range[i], 
                                                                                bins_per_edge=bins_per_edge)

    n_clusters += [np.amax(data_by_cluster)+1]
    print(n_clusters)

    # ####### PLOTTING DENSITY MAP BY CONDITION AND AT TIMEPOINT 1.25#######
    condition_ID = metadata['Condition'].values.tolist()
    print("PLOTTING DENSITY MAP BY CONDITION AND AT TIMEPOINT 1.25")
    x_range = int(np.ceil(np.amax(final_embedding[:,0])) - np.floor(np.amin(final_embedding[:,0])))
    y_range = int(np.ceil(np.amax(final_embedding[:,1])) - np.floor(np.amin(final_embedding[:,1])))
    ax_range = [[int(np.floor(np.amin(final_embedding[:,0]))-x_range/40),int(np.ceil(np.amax(final_embedding[:,0]))+x_range/40)],
                [int(np.floor(np.amin(final_embedding[:,1]))-y_range/40),int(np.ceil(np.amax(final_embedding[:,1]))+y_range/40)]]


    ####### SUBSETTING AND REORGANIZING DATA FOR DOWNSTREAM ANALYSIS ########
    comparison = ['Baseline','Lesion2']
    an_embedding = final_embedding[metadata.index[(metadata['Condition'].isin(condition_names)) &
                                                (metadata['Timepoint']==1.25)].to_list(),:]
    an_data_by_cluster = data_by_cluster[metadata.index[(metadata['Condition'].isin(condition_names)) &
                                                (metadata['Timepoint']==1.25)].to_list()]
    an_meta = metadata.loc[(metadata['Condition'].isin(condition_names)) &
                        (metadata['Timepoint']==1.25)].reset_index() # Expanded meta for each data point
    an_meta = an_meta.sort_values(['Condition','AnimalID'])
    an_embedding = an_embedding[an_meta.index.to_list(),:]
    an_data_by_cluster = an_data_by_cluster[an_meta.index.to_list()]
    an_meta_vid = meta_vid.loc[(meta_vid['Condition'].isin(condition_names)) &
                            (meta_vid['Timepoint']==1.25)].reset_index() # Meta for each video
    an_meta_vid = an_meta_vid.sort_values(['Condition','AnimalID'])

    # import pdb; pdb.set_trace()
    print("Obtaining density maps for each video")

    ####### Calculating cluster percentage by batchID/recordingID #######
    cluster_freq_by_vid = cluster_frequencies(an_data_by_cluster, an_meta['batchID'].values)

    # First index of occurence of condition (assuming sorted)
    sorted_condition_idx = [an_meta_vid['Condition'].to_list().index(condition) for condition in condition_names]
    from sklearn.metrics.pairwise import cosine_similarity
    cos_by_vid = cosine_similarity(cluster_freq_by_vid)

    freq_shuffled = np.empty((0,cluster_freq_by_vid.shape[1]))
    for i in range(cluster_freq_by_vid.shape[0]):
        permutation = np.expand_dims(np.random.permutation(cluster_freq_by_vid[i,:]),axis=0)
        freq_shuffled = np.append(freq_shuffled,permutation,axis=0)
        
    cos_neg_ctrl = cosine_similarity(freq_shuffled)

    sorted_condition_idx+=[24]

    cos_idx_base = (sorted_condition_idx[condition_names.index(comparison[0])],
                 sorted_condition_idx[condition_names.index(comparison[0])+1])
    cos_idx_pd = (sorted_condition_idx[condition_names.index(comparison[1])],
                 sorted_condition_idx[condition_names.index(comparison[1])+1])


    cos_sim_subset = cos_by_vid[cos_idx_base[0]:cos_idx_base[1],cos_idx_pd[0]:cos_idx_pd[1]]
    all_cos_mean += [np.mean(cos_sim_subset)]
    all_cos_se += [np.std(cos_sim_subset)/np.sqrt(np.size(cos_sim_subset))]

    # import pdb; pdb.set_trace()

    cos_inter_base = cos_by_vid[cos_idx_base[0]:cos_idx_base[1],cos_idx_base[0]:cos_idx_base[1]]
    cos_inter_base = cos_inter_base[np.triu_indices(cos_inter_base.shape[0],1)]
    base_cos_mean +=[np.mean(cos_inter_base)]
    base_cos_se +=[np.std(cos_inter_base)/np.sqrt(np.size(cos_inter_base))]

    cos_inter_pd = cos_by_vid[cos_idx_pd[0]:cos_idx_pd[1],cos_idx_pd[0]:cos_idx_pd[1]]
    cos_inter_pd = cos_inter_pd[np.triu_indices(cos_inter_pd.shape[0],1)]
    pd_cos_mean +=[np.mean(cos_inter_pd)]
    pd_cos_se +=[np.std(cos_inter_pd)/np.sqrt(np.size(cos_inter_pd))]

    cos_neg = cos_neg_ctrl[np.triu_indices(cos_neg_ctrl.shape[0],1)]
    neg_cos_mean += [np.mean(cos_neg)]
    neg_cos_se += [np.std(cos_neg)/np.sqrt(np.size(cos_neg))]


    if i==0:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(watershed_borders[:,0],watershed_borders[:,1],'.r',markersize=0.1)
    
        embedding_by_ID = final_embedding[metadata.index[(metadata['Condition']=='Baseline') & 
                                                        (metadata['Timepoint']==1.25)].to_list(),:]
        density_by_ID, _, _ = map_density(embedding_by_ID,bins_per_edge=bins_per_edge,sigma=best_sigma, max_clip=0.75, x_range=x_range,y_range=y_range, hist_range=ax_range)
        ax.imshow(density_by_ID)
        ax.set_aspect('auto')
        filename = ''.join([plot_folder,'density_0_base','.png'])
        plt.savefig(filename,dpi=400)
        plt.close()


        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(watershed_borders[:,0],watershed_borders[:,1],'.r',markersize=0.1)
        embedding_by_ID = final_embedding[metadata.index[(metadata['Condition']=='Lesion2') & 
                                                        (metadata['Timepoint']==1.25)].to_list(),:]
        density_by_ID, _, _ = map_density(embedding_by_ID,bins_per_edge=bins_per_edge,sigma=best_sigma, max_clip=0.75, x_range=x_range,y_range=y_range, hist_range=ax_range)
        ax.imshow(density_by_ID)

        ax.set_aspect('auto')
        filename = ''.join([plot_folder,'density_0_pd','.png'])
        plt.savefig(filename,dpi=400)
        plt.close()
    elif i==len(param_range)-1:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(watershed_borders[:,0],watershed_borders[:,1],'.r',markersize=0.1)
    
        embedding_by_ID = final_embedding[metadata.index[(metadata['Condition']=='Baseline') & 
                                                        (metadata['Timepoint']==1.25)].to_list(),:]
        density_by_ID, _, _ = map_density(embedding_by_ID,bins_per_edge=bins_per_edge,sigma=best_sigma, max_clip=0.75, x_range=x_range,y_range=y_range, hist_range=ax_range)
        ax.imshow(density_by_ID)
        ax.set_aspect('auto')
        filename = ''.join([plot_folder,'density_max_base','.png'])
        plt.savefig(filename,dpi=400)
        plt.close()


        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(watershed_borders[:,0],watershed_borders[:,1],'.r',markersize=0.1)
        embedding_by_ID = final_embedding[metadata.index[(metadata['Condition']=='Lesion2') & 
                                                        (metadata['Timepoint']==1.25)].to_list(),:]
        density_by_ID, _, _ = map_density(embedding_by_ID,bins_per_edge=bins_per_edge,sigma=best_sigma, max_clip=0.75, x_range=x_range,y_range=y_range, hist_range=ax_range)
        ax.imshow(density_by_ID)

        ax.set_aspect('auto')
        filename = ''.join([plot_folder,'density_max_pd','.png'])
        plt.savefig(filename,dpi=400)
        plt.close()


f = plt.figure()
plt.errorbar(range(len(all_cos_mean)), all_cos_mean, 
             marker='s', linewidth=0, elinewidth=1, 
             yerr=1.96*np.array(all_cos_se), label='Healthy vs PD')

plt.errorbar(range(len(all_cos_mean)), base_cos_mean, 
             marker='s', linewidth=0, elinewidth=1, 
             yerr=1.96*np.array(base_cos_se), label='W/in Healthy')

plt.errorbar(range(len(all_cos_mean)), pd_cos_mean, 
             marker='s', linewidth=0, elinewidth=1, 
             yerr=1.96*np.array(pd_cos_se), label='W/in PD')

plt.errorbar(range(len(all_cos_mean)), neg_cos_mean, 
             marker='s', linewidth=0, elinewidth=1, 
             yerr=1.96*np.array(neg_cos_se), label='Shuffle')

plt.legend()
plt.ylabel('Cosine Similarity')

plt.xticks(ticks=range(len(all_cos_mean)), labels=param_range)#, rotation='vertical')
# plt.xlabel('t-SNE Perplexity')
# plt.xlabel('t-SNE Number of Neighbors')
plt.xlabel('$\sigma$')
# plt.xlabel('k-Nearest Neighbors')
plt.savefig(''.join([plot_folder,'base_pd_cos.png']),dpi=400)

plt.close()

f = plt.figure()
plt.plot(range(len(all_cos_mean)),np.log(n_clusters), marker='s', linewidth=0)
plt.ylabel('# Clusters')

plt.xticks(ticks=range(len(all_cos_mean)), labels=param_range)
# plt.xlabel('t-SNE Number of Neighbors')
# plt.xlabel('t-SNE Perplexity')
# plt.xlabel('k-Nearest Neighbors')
plt.xlabel('$\sigma$')
plt.savefig(''.join([plot_folder,'n_clusters.png']),dpi=400)
plt.close()


