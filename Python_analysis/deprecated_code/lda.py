from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

def sample_rand_columns(x, n=1000):
    ix_i = np.random.sample(x.shape).argsort(axis=0)[:n,:]
    ix_j = np.tile(np.arange(x.shape[1]), (n, 1))
    return x[ix_i, ix_j]

config_file = sys.argv[1]
params = load_data.read_config(config_file)

plot_folder = ''.join([params['out_folder'],'lda_analysis/'])

os.makedirs(plot_folder,exist_ok=True)
[feats, batch_ID] = load_data.load_data(params['analysis_path'],
                                           params['preds_path'],
                                           params['batch_name'],
                                           subsample=1)
# feats = hdf5storage.loadmat('../CAPTURE_data/48_tadross_data/jt_features_fixed_conservative.mat')['jt_features_fixed_conservative']
feats = hdf5storage.loadmat('../CAPTURE_data/48_tadross_data/jt_features_fixed_recovered.mat')['jt_features_fixed_recovered']
# print("jt_features_fixed_conservative: ",feats.shape)
feats[0,:] = 0

meta_by_vid = pd.read_csv(params['meta_path'])[['Condition','AnimalID','Timepoint']]
meta_by_frame = meta_by_vid.iloc[batch_ID].reset_index().rename(columns={'index':'batchID'})
condition_names = ['Baseline','Lesion2','LDOPA','Rx-iSPN']

condition_idx_by_frame = (meta_by_frame['Condition'].isin(condition_names)) & (meta_by_frame['Timepoint']==1.25)

feats = feats[meta_by_frame.index[condition_idx_by_frame],:]
batch_ID = batch_ID[meta_by_frame.index[condition_idx_by_frame]]

meta_by_frame = meta_by_frame.loc[condition_idx_by_frame].reset_index()
meta_by_vid = meta_by_vid.loc[condition_idx_by_frame].reset_index().rename(columns={'index':'vidID'})

labels_base_pd = np.array([1 if condition=='Baseline' else 0 for condition in meta_by_frame['Condition'].values])
labels_ldopa_rx = np.array([1 if condition=='Rx-iSPN' else 0 for condition in meta_by_frame['Condition'].values])
animal_ID = meta_by_frame['AnimalID'].values
rx_animals = animal_ID[meta_by_frame['Condition']=='Rx-iSPN']
animals = np.unique(rx_animals)#animal_ID)
n_bins = 100


clf = LinearDiscriminantAnalysis(n_components=1)
feats_rx_animals = feats[meta_by_frame['AnimalID'].isin(rx_animals)]
rx_animal_meta = meta_by_frame.loc[meta_by_frame['AnimalID'].isin(rx_animals)]

## Removing outlier features
clf.fit(feats_rx_animals[rx_animal_meta['Condition'].isin(['Baseline','Lesion2'])],
        labels_base_pd[meta_by_frame['Condition'].isin(['Baseline','Lesion2']) & meta_by_frame['AnimalID'].isin(rx_animals)])
lda_coeffs = clf.coef_
good_fx = ((lda_coeffs<15) & (lda_coeffs>-15)).nonzero()
feats = feats[:,good_fx[1]]
feats_rx_animals = feats_rx_animals[:,good_fx[1]]
clf.fit(feats_rx_animals[rx_animal_meta['Condition'].isin(['LDOPA','Rx-iSPN'])],
        labels_ldopa_rx[meta_by_frame['Condition'].isin(['LDOPA','Rx-iSPN']) & meta_by_frame['AnimalID'].isin(rx_animals)])
lda_coeffs = clf.coef_
good_fx = ((lda_coeffs<15) & (lda_coeffs>-15)).nonzero()
feats_rx_animals = feats_rx_animals[:,good_fx[1]]
feats = feats[:,good_fx[1]]

## Refitting with new features
clf.fit(feats_rx_animals[rx_animal_meta['Condition'].isin(['Baseline','Lesion2'])],
        labels_base_pd[meta_by_frame['Condition'].isin(['Baseline','Lesion2']) & meta_by_frame['AnimalID'].isin(rx_animals)])
lda_all = clf.transform(feats_rx_animals)
lda_coeffs = clf.coef_

clf.fit(feats_rx_animals[rx_animal_meta['Condition'].isin(['LDOPA','Rx-iSPN'])],
        labels_ldopa_rx[meta_by_frame['Condition'].isin(['LDOPA','Rx-iSPN']) & meta_by_frame['AnimalID'].isin(rx_animals)])
lda_all = np.append(lda_all,clf.transform(feats_rx_animals),axis=1)
lda_coeffs = np.append(lda_coeffs,clf.coef_,axis=0)
print(np.shape(lda_coeffs))

lda_baseline = np.squeeze(lda_all[rx_animal_meta['Condition']=='Baseline',:])
lda_pd = np.squeeze(lda_all[rx_animal_meta['Condition']=='Lesion2',:])
lda_ldopa = np.squeeze(lda_all[rx_animal_meta['Condition']=='LDOPA',:])
lda_rx = np.squeeze(lda_all[rx_animal_meta['Condition']=='Rx-iSPN',:])

fig = plt.figure(figsize=(9,9))
gs = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)

ax_main.scatter(lda_baseline[:,0],lda_baseline[:,1], s=1, alpha=0.25, label='Healthy')
ax_main.scatter(lda_pd[:,0],lda_pd[:,1], s=1, alpha=0.25, label='Lesion')
ax_main.scatter(lda_ldopa[:,0],lda_ldopa[:,1], s=1, alpha=0.25, label='LDOPA')
ax_main.scatter(lda_rx[:,0],lda_rx[:,1], s=1, alpha=0.25, label='Rx-iSPN')
ax_main.set_xlim(-5,5)
ax_main.set_ylim(-5,5)
ax_main.legend()
ax_main.set_xlabel('Lesion -> Healthy')
ax_main.set_ylabel('LDOPA -> Rx-iSPN')

ax_xDist.hist([lda_baseline[:,0],lda_pd[:,0],lda_ldopa[:,0],lda_rx[:,0]], 
                bins=n_bins, range=(-5,5), align='mid',
                label=['Healthy','Lesion','LDOPA','Rx-iSPN'])
ax_yDist.hist([lda_baseline[:,1],lda_pd[:,1],lda_ldopa[:,1],lda_rx[:,1]], 
                bins=n_bins, range=(-5,5), align='mid', orientation='horizontal',
                label=['Healthy','Lesion','LDOPA','Rx-iSPN'])

plt.savefig(''.join([plot_folder,'Rx_iSPN_lda_all.png']),dpi=400)
plt.close()

f=plt.figure(figsize=(8,8))
plt.scatter(lda_coeffs[0,:],lda_coeffs[1,:])
plt.xlabel('Lesion -> Baseline LDA Coefficients')
plt.ylabel('LDOPA -> Rx-iSPN LDA Coefficients')
# plt.xlim(-20,20)
# plt.ylim(-20,20)
plt.savefig(''.join([plot_folder,'lda_coeffs.png']),dpi=400)
plt.close()


# #### CDF OF FEATURES
# k = 200
# n = 1000
# clf = LinearDiscriminantAnalysis(n_components=1)
# feats_base = feats[meta_by_frame['Condition']=='Baseline',:]
# feats_pd = feats[meta_by_frame['Condition']=='Lesion2',:]
# print(feats_base.shape)
# print(feats_pd.shape)
# d_arr=np.empty((0,feats_base.shape[1]))
# for i in tqdm(range(k)):
#     # import pdb; pdb.set_trace()
#     base_sample = np.zeros((n,feats_base.shape[1]))
#     pd_sample = np.zeros((n,feats_base.shape[1]))
#     for j in range(feats_base.shape[1]):
#         base_sample[:,j] = np.random.choice(feats_base[:,j],size=n,replace=False)
#         pd_sample[:,j] = np.random.choice(feats_pd[:,j],size=n,replace=False)
#     # base_sample = sample_rand_columns(feats_base)
#     # pd_sample = sample_rand_columns(feats_pd)

#     d_p = np.absolute(np.mean(base_sample,axis=0)-np.mean(pd_sample,axis=0))
#     d_p = d_p*2/(np.std(base_sample,axis=0)+np.std(pd_sample,axis=0))
#     d_arr = np.append(d_arr,np.expand_dims(d_p,axis=0),axis=0)
# print(np.amin(d_arr))
# print(np.amax(d_arr))
# # import pdb; pdb.set_trace()
# f = plt.figure(figsize=(8,8))
# for i in tqdm(range(d_arr.shape[1])):
#     import pdb; pdb.set_trace()
#     plt.hist(d_arr[:,i],bins=1000,density=True,range=(0,np.amax(d_arr)*2),cumulative=True,histtype="step")

# plt.savefig(''.join([plot_folder,'feats_cdf.png']),dpi=400)

# import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()
for i in range(len(animals)):
    clf_base_pd = LinearDiscriminantAnalysis(n_components=1)
    sub_labels_base_pd = labels_base_pd[(animal_ID==animals[i]) & meta_by_frame['Condition'].isin(['Baseline','Lesion2'])]
    control_feats = feats[(animal_ID==animals[i]) & meta_by_frame['Condition'].isin(['Baseline','Lesion2']),:]
    clf_base_pd.fit(control_feats,sub_labels_base_pd)
    lda_feats = clf_base_pd.transform(feats[animal_ID==animals[i]])

    # import pdb; pdb.set_trace()
    sub_labels_ldopa_rx = labels_ldopa_rx[(animal_ID==animals[i]) & meta_by_frame['Condition'].isin(['Rx-iSPN','LDOPA'])]
    control_feats = feats[(animal_ID==animals[i]) & meta_by_frame['Condition'].isin(['Rx-iSPN','LDOPA']),:]
    clf_ldopa_rx = LinearDiscriminantAnalysis(n_components=1)
    clf_ldopa_rx.fit(control_feats,sub_labels_ldopa_rx)
    lda_feats = np.append(lda_feats, clf_ldopa_rx.transform(feats[animal_ID==animals[i]]),axis=1)

    animal_meta = meta_by_frame.loc[meta_by_frame.index[animal_ID == animals[i]]]

    lda_baseline = np.squeeze(lda_feats[animal_meta['Condition']=='Baseline',:])
    lda_pd = np.squeeze(lda_feats[animal_meta['Condition']=='Lesion2',:])
    lda_ldopa = np.squeeze(lda_feats[animal_meta['Condition']=='LDOPA',:])
    lda_rx = np.squeeze(lda_feats[animal_meta['Condition']=='Rx-iSPN',:])
    print(lda_baseline.shape)
    print(lda_pd.shape)
    print(lda_ldopa.shape)
    print(lda_rx.shape)

    hist_min_x = np.amin(lda_feats[:,0])
    hist_max_x = np.amax(lda_feats[:,0])
    hist_min_y = np.amin(lda_feats[:,1])
    hist_max_y = np.amax(lda_feats[:,1])

    fig = plt.figure(figsize=(9,9))
    gs = gridspec.GridSpec(3, 3)
    ax_main = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)

    ax_main.scatter(lda_baseline[:,0],lda_baseline[:,1], s=1, alpha=0.25, label='Healthy')
    ax_main.scatter(lda_pd[:,0],lda_pd[:,1], s=1, alpha=0.25, label='Lesion')
    ax_main.scatter(lda_ldopa[:,0],lda_ldopa[:,1], s=1, alpha=0.25, label='LDOPA')
    ax_main.scatter(lda_rx[:,0],lda_rx[:,1], s=1, alpha=0.25, label='Rx-iSPN')
    ax_main.set_xlim(-5,5)
    ax_main.set_ylim(-5,5)
    ax_main.legend()
    ax_main.set_xlabel('Lesion -> Healthy')
    ax_main.set_ylabel('LDOPA -> Rx-iSPN')

    ax_xDist.hist([lda_baseline[:,0],lda_pd[:,0],lda_ldopa[:,0],lda_rx[:,0]], 
                    bins=n_bins, range=(-5,5), align='mid',
                    label=['Healthy','Lesion','LDOPA','Rx-iSPN'])
    ax_yDist.hist([lda_baseline[:,1],lda_pd[:,1],lda_ldopa[:,1],lda_rx[:,1]], 
                    bins=n_bins, range=(-5,5), align='mid', orientation='horizontal',
                    label=['Healthy','Lesion','LDOPA','Rx-iSPN'])
    # import pdb; pdb.set_trace()

    plt.savefig(''.join([plot_folder,'2xlda_anml_',str(animals[i]),'.png']),dpi=400)
    plt.close()



