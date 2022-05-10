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

plot_folder = ''.join([params['out_folder'],'final_figures/'])

###### LOAD TEMPLATE EMBEDDING DATA ######
results_path = ''.join([params['out_folder'],'/results.mat'])
results = hdf5storage.loadmat(results_path, variable_names=['template'])#, 'template_embedding'])
template = results['template']
print(np.shape(template))
# template_embedding = results['template_embedding']

###### COMPLEXITY OF TEMPLATE EMBEDDING ######
n = np.shape(template)[0]
import tsnecuda as tc
tsne = tc.TSNE(n_iter=2500, verbose=2, num_neighbors=300, perplexity=70, learning_rate=int(n/12))
template_embedding = tsne.fit_transform(template)
filename = ''.join([plot_folder, 'template'])
embed_scatter(template_embedding, filename=filename)
clustering(template_embedding, filename)

# import pdb; pdb.set_trace()

###### 10-FOLD CROSS VALIDATION FOR # NEAREST NEIGHBORS ######
# min_err_nn = k_fold_embed(template, template_embedding,
#                    k_split=10, nn_range=list(range(1,16,1)),
#                    plot_folder=plot_folder)

k_fold_embed(template,template_embedding, k_split=10, nn_range=[5], plot_folder=plot_folder)
k_fold_embed(template,template_embedding, k_split=10, nn_range=[20], plot_folder=plot_folder)

# import pdb; pdb.set_trace()

###### LOAD FULL DATA ######
subsample_rate = 30
[features, batch_ID] = load_data.load_data(params['analysis_path'],params['preds_path'],params['batch_name'],subsample=subsample_rate)


###### VALIDATING CLUSTER NUMBERS/SIGMA WITH AIC ######
## Getting final reembedding using nn obtained from kfold
min_err_nn=5 
print("Obtaining final embedding")
from KNNEmbed import KNNEmbed
start = time.time()
reembedder = KNNEmbed(k=min_err_nn)
reembedder.fit(template,template_embedding)
final_embedding = reembedder.predict(features, weights='distance')
print("Total Time ReEmbedding: ", time.time()-start)


save_file = {'final_embedding':final_embedding}
hdf5storage.savemat(''.join([plot_folder,'final_embedding.mat']), save_file)
print("Saving to ", ''.join([plot_folder,'final_embedding.mat']))

final_embedding = hdf5storage.loadmat(''.join([plot_folder,'final_embedding.mat']), variable_names=['final_embedding'])['final_embedding']
# import pdb; pdb.set_trace()

## Finding best cluster using AIC
bins_per_edge = 5000
sigma_range = list(range(25,80,5))
# validate_cluster_num(final_embedding, 
#                      bins_per_edge=bins_per_edge, 
#                      sigma_range=sigma_range,
#                      plot_folder=plot_folder,
#                      metric='explained_variance')

# import pdb; pdb.set_trace()

best_sigma = 45
filename = ''.join([plot_folder, 'final_embed'])
embed_scatter(final_embedding, filename=filename)
watershed_map, watershed_borders, data_by_cluster, density_map = clustering(final_embedding, filename, sigma=best_sigma, bins_per_edge=bins_per_edge)

###### LOAD METADATA FOR DATA ######
meta_vid = pd.read_csv(params['meta_path'])[['Condition','AnimalID','Timepoint']]
# animal_vid = pd.read_csv(params['meta_path'])[['AnimalID']]
metadata = meta_vid.iloc[batch_ID].reset_index().rename(columns={'index':'batchID'})
# import pdb; pdb.set_trace()



# ####### PLOTTING DENSITY MAP BY CONDITION AND AT TIMEPOINT 1.25#######
condition_ID = metadata['Condition'].values.tolist()
print("PLOTTING DENSITY MAP BY CONDITION AND AT TIMEPOINT 1.25")
x_range = int(np.ceil(np.amax(final_embedding[:,0])) - np.floor(np.amin(final_embedding[:,0])))
y_range = int(np.ceil(np.amax(final_embedding[:,1])) - np.floor(np.amin(final_embedding[:,1])))
ax_range = [[int(np.floor(np.amin(final_embedding[:,0]))-x_range/40),int(np.ceil(np.amax(final_embedding[:,0]))+x_range/40)],
            [int(np.floor(np.amin(final_embedding[:,1]))-y_range/40),int(np.ceil(np.amax(final_embedding[:,1]))+y_range/40)]]

os.makedirs(''.join([plot_folder,'density_by_condition/']),exist_ok=True)
condition_names = ['Baseline','LDOPA','Lesion2','Rx-iSPN']
for batch in condition_names:
    # import pdb; pdb.set_trace()
    embedding_by_ID = final_embedding[metadata.index[(metadata['Condition']==batch) & 
                                                     (metadata['Timepoint']==1.25)].to_list(),:]
    # embedding_by_ID = embedding_by_ID[metadata.index[metadata['Timepoint']=='1.25'].to_list(),:]
    density_by_ID, _, _ = map_density(embedding_by_ID,bins_per_edge=bins_per_edge,sigma=best_sigma, max_clip=0.75, x_range=x_range,y_range=y_range, hist_range=ax_range)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(watershed_borders[:,0],watershed_borders[:,1],'.r',markersize=0.05)
    ax.imshow(density_by_ID)
    ax.set_aspect('auto')
    filename = ''.join([plot_folder,'density_by_condition/density_vid_',str(batch),'.png'])
    plt.savefig(filename,dpi=400)
    plt.close()


####### SUBSETTING AND REORGANIZING DATA FOR DOWNSTREAM ANALYSIS ########
comparisons = [('Baseline','Lesion2'),('Baseline','LDOPA'),('Baseline','Rx-iSPN')]
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
                           (meta_vid['Timepoint']==1.25)].reset_index().rename(columns={'index':'vidID'}) # Meta for each video
an_meta_vid = an_meta_vid.sort_values(['Condition','AnimalID'])
####### Obtaining density maps for each video #######
# import pdb; pdb.set_trace()

print("Obtaining density maps for each video")
vid_ID = an_meta_vid['vidID'].tolist()
densities = np.zeros((len(vid_ID),1000**2)) #video x flattened density map
for i,video in enumerate(vid_ID):
    # import pdb; pdb.set_trace()
    embedding_by_vid = an_embedding[an_meta.index[an_meta['batchID']==video].to_list(),:]
    density_by_vid, _, _ = map_density(embedding_by_vid,
                                       bins_per_edge=1000,
                                       sigma=15, 
                                       max_clip=1, 
                                       x_range=x_range,
                                       y_range=y_range, 
                                       hist_range=ax_range)
    densities[i,:] = (density_by_vid/np.sum(density_by_vid)).flatten()


####### Calculating cluster percentage by batchID/recordingID #######
cluster_freq_by_vid = cluster_frequencies(an_data_by_cluster, an_meta['batchID'].values)

# First index of occurence of condition (assuming sorted)
sorted_condition_idx = [an_meta_vid['Condition'].to_list().index(condition) for condition in condition_names]
from sklearn.metrics.pairwise import cosine_similarity
cos_sim_by_vid = cosine_similarity(cluster_freq_by_vid)
f = plt.figure()
plt.imshow(cos_sim_by_vid)
plt.xticks(sorted_condition_idx, condition_names)
plt.yticks(sorted_condition_idx, condition_names)
plt.colorbar()
plt.savefig(''.join([plot_folder,'cos_sim_matrix.png']),dpi=400)
plt.close()


mean_pair_cos, se_pair_cos = [], []
mean_pair_kl, se_pair_kl = [], []
for comparison in comparisons:
    group1_meta = an_meta_vid[an_meta_vid['Condition']==comparison[0]]
    group2_meta = an_meta_vid[an_meta_vid['Condition']==comparison[1]]
    combined_animal_ids = list(set(group1_meta['AnimalID'].to_list()) & 
                               set(group2_meta['AnimalID'].to_list()))

    group1_idx = an_meta_vid.index[(an_meta_vid['Condition']==comparison[0]) &
                                 (an_meta_vid['AnimalID'].isin(combined_animal_ids))].to_list()
    group2_idx = an_meta_vid.index[(an_meta_vid['Condition']==comparison[1]) &
                                 (an_meta_vid['AnimalID'].isin(combined_animal_ids))].to_list()

    group1_densities = densities[group1_idx,:]
    group2_densities = densities[group2_idx,:]
    n_animals = len(group1_idx)
    pair_cos = np.zeros(n_animals)
    pair_kl = np.zeros(n_animals)
    for i in range(n_animals):
        pair_cos[i] = distance.cosine(group1_densities[i,:], group2_densities[i,:])
        pair_kl[i] = np.sum(kl_div(np.clip(group2_densities[i,:],1e-50,None),
                                   np.clip(group1_densities[i,:],1e-50,None)))

    mean_pair_cos += [np.mean(pair_cos)]
    se_pair_cos += [np.std(pair_cos)/np.sqrt(n_animals)]

    mean_pair_kl += [np.mean(pair_kl)]
    se_pair_kl += [np.std(pair_kl)/np.sqrt(n_animals)]

f = plt.figure(figsize=(8,10))
plt.errorbar(np.arange(3), mean_pair_cos, 
             marker='s', linewidth=0, elinewidth=1, 
             yerr=1.96*np.array(se_pair_cos))

plt.xticks(ticks=np.arange(3), labels=['Lesion2','LDOPA','Rx-iSPN'], rotation='vertical')
plt.xlabel('Condition')
plt.ylabel('Pairwise Cosine Similarity with Healthy')
plt.savefig(''.join([plot_folder,'pair_cos.png']),dpi=400)

f = plt.figure(figsize=(8,10))
plt.errorbar(np.arange(3), mean_pair_kl, 
             marker='s', linewidth=0, elinewidth=1, 
             yerr=1.96*np.array(se_pair_kl))

plt.xticks(ticks=np.arange(3), labels=['Lesion2','LDOPA','Rx-iSPN'], rotation='vertical')
plt.xlabel('Condition')
plt.ylabel('Pairwise KL Divergence with Healthy')
plt.savefig(''.join([plot_folder,'pair_kl_div.png']),dpi=400)

# f_cos = plt.figure()
cos_sim_comp_mean = []
cos_sim_comp_se = []
sorted_condition_idx+=[40]
for comparison in comparisons:
    groups, mean_groups, se_groups = [], [], []
    for i in range(2):
        groups += [cluster_freq_by_vid[an_meta_vid.index[an_meta_vid['Condition']==comparison[i]].to_list(),:]]
        mean_groups += [np.mean(groups[i], axis=0)]
        n = groups[i].shape[0]
        se_groups += [np.clip(np.std(groups[i], axis=0)/np.sqrt(n),1e-6,None)]

    ### Paired t test ###
    group1_meta = an_meta_vid[an_meta_vid['Condition']==comparison[0]]
    group2_meta = an_meta_vid[an_meta_vid['Condition']==comparison[1]]
    combined_animal_ids = list(set(group1_meta['AnimalID'].to_list()) & 
                               set(group2_meta['AnimalID'].to_list()))

    group1_idx = an_meta_vid.index[(an_meta_vid['Condition']==comparison[0]) &
                                 (an_meta_vid['AnimalID'].isin(combined_animal_ids))].to_list()
    group2_idx = an_meta_vid.index[(an_meta_vid['Condition']==comparison[1]) &
                                 (an_meta_vid['AnimalID'].isin(combined_animal_ids))].to_list()

    group_animals1 = cluster_freq_by_vid[group1_idx, :]
    group_animals2 = cluster_freq_by_vid[group2_idx, :]
    diff = group_animals2 - group_animals1
    paired_t = np.mean(diff,axis=0)/np.clip((np.std(diff,axis=0)/np.sqrt(len(group2_idx))),1e-20,None)

    if comparison == ('Baseline','Lesion2'):
        # t = (mean_groups[0]-mean_groups[1])#/np.sqrt(std_groups[0]**2/groups[0].shape[0]+std_groups[1]**2/groups[1].shape[0])
        sorted_t_idx = np.argsort(paired_t)
        idx_sig = np.array((paired_t[sorted_t_idx]>=1.96) | (paired_t[sorted_t_idx]<=-1.96)).nonzero()[0]
        sorted_t_idx = sorted_t_idx[idx_sig]
        # sorted_t_idx = np.append(sorted_t_idx[:20], sorted_t_idx[-20:])
    
    paired_t = paired_t[sorted_t_idx]
    idx_sig = np.array((paired_t>=1.96) | (paired_t<=-1.96)).nonzero()[0]
    idx_insig = np.array((paired_t<1.96) & (paired_t>-1.96)).nonzero()[0]



    f = plt.figure(figsize=(16,8))
    colors = ['k','r']
    for i in range(2):
        mean_groups_sort = mean_groups[i][sorted_t_idx]
        se_groups_sort = se_groups[i][sorted_t_idx]
        if mean_groups_sort[idx_sig] is not None:
            plt.errorbar(idx_sig+i*0.1, mean_groups_sort[idx_sig], 
                    marker='v', linewidth=0, elinewidth=1, yerr=1.96*se_groups_sort[idx_sig],
                    color=colors[i], label=comparison[i])

        if mean_groups_sort[idx_insig] is not None:
            plt.errorbar(idx_insig+i*0.1, mean_groups_sort[idx_insig], 
                    marker='s', linewidth=0, elinewidth=1, yerr=1.96*se_groups_sort[idx_insig],
                    color=colors[i], label=comparison[i])
    
    # legend1 = plt.legend([l1,l2],['Paired T Significant','Paired T Insignificant'],loc=2)
    plt.legend()
    # plt.gca().add_artist(legend1)
    plt.xticks(ticks=np.arange(len(sorted_t_idx)), labels=sorted_t_idx)
    plt.xlabel('Cluster')
    plt.ylabel('Behavior Usage')
    plt.savefig(''.join([plot_folder,comparison[0],'_',comparison[1],'_compare.png']),dpi=400)
    
    cos_sim_idx_r = (sorted_condition_idx[condition_names.index(comparison[0])],
                     sorted_condition_idx[condition_names.index(comparison[0])+1])
    cos_sim_idx_c = (sorted_condition_idx[condition_names.index(comparison[1])],
                       sorted_condition_idx[condition_names.index(comparison[1])+1])

    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    cos_sim_subset = cos_sim_by_vid[cos_sim_idx_r[0]:cos_sim_idx_r[1],cos_sim_idx_c[0]:cos_sim_idx_c[1]]
    cos_sim_comp_mean += [np.mean(cos_sim_subset)]
    cos_sim_comp_se += [np.std(cos_sim_subset)/np.sqrt(np.size(cos_sim_subset))]

f = plt.figure(figsize=(8,10))
plt.errorbar(np.arange(3), cos_sim_comp_mean, 
             marker='s', linewidth=0, elinewidth=1, 
             yerr=1.96*np.array(cos_sim_comp_se))

plt.xticks(ticks=np.arange(3), labels=['Lesion2','LDOPA','Rx-iSPN'], rotation='vertical')
plt.xlabel('Condition')
plt.ylabel('Cosine Similarity with Healthy')
plt.savefig(''.join([plot_folder,'avg_cos_sim.png']),dpi=400)



