import numpy as np
import matplotlib.pyplot as plt
from KNNEmbed import KNNEmbed
import time
from plots import *

def k_fold_embed(data, predictions, k_split=10, nn_range=list(range(1,22,2)), plot_folder='./plots/', metric='mse'):

    print("Embedding 10-fold data")
    print(data.shape)
    print(predictions.shape)
    data_shuffled = np.random.permutation(np.append(predictions, data, axis=1))
    split_size = np.floor(data_shuffled.shape[0]/k_split)
    max_dist = np.sqrt(np.sum((np.amax(predictions,axis=0)-np.amin(predictions,axis=0))**2))
    metric_vals, min_metric_embedding = [], []
    for nn in nn_range:
        print("Reembedding with ",nn," nearest neighbors")
        nn_pred_embedding = np.empty((0,2))
        start = time.time()
        for i in range(k_split):
            train = np.delete(data_shuffled, range(int(i*split_size), int((i+1)*split_size)), axis=0)
            test = data_shuffled[int(i*split_size) : int((i+1)*split_size), :]
            
            reembedder = KNNEmbed(k=nn)
            reembedder.fit(train[:,2:],train[:,0:2])
            nn_pred_embedding = np.append(nn_pred_embedding,
                                        reembedder.predict(test[:,2:],weights='distance'),
                                        axis=0)
        print("Total Time K-Fold Reembedding: ", time.time()-start)
        
        # import pdb; pdb.set_trace()
        if metric == 'euclidean':

            curr_metric = np.mean(np.sqrt(np.sum((data_shuffled[:nn_pred_embedding.shape[0],:2] - nn_pred_embedding)**2,axis=1)))
        elif metric == 'mse':
            curr_metric = np.mean(np.sum((data_shuffled[:nn_pred_embedding.shape[0],:2] - nn_pred_embedding)**2,axis=1))

        curr_metric = curr_metric/max_dist
        print(curr_metric)
            
        print("Reembedding Metric: ", curr_metric)
        # if metric is empty or curr_metric is lowest so far
        if not metric_vals or all(curr_metric<val for val in metric_vals):
            min_metric_embedding = nn_pred_embedding
            min_metric_nn = nn
    
        metric_vals += [curr_metric]


    f = plt.figure()
    plt.scatter(predictions[:,0], predictions[:,1], marker='.', s=3, linewidths=0,
                c='b', label='Targets')
    plt.scatter(min_metric_embedding[:,0], min_metric_embedding[:,1], marker='.', s=3, linewidths=0,
                c='m', label='CV Predictions')
    plt.legend()
    plt.savefig(''.join([plot_folder,'k_fold_mbed_',str(min_metric_nn),'nn.png']), dpi=400)
    plt.close()
        
    f = plt.figure()
    plt.plot(nn_range, metric_vals, marker='o', c='k')
    plt.savefig(''.join([plot_folder,'k_fold_embed_metric.png']), dpi=400)
    plt.close()

    return min_metric_nn

def validate_cluster_num(data_embedding, bins_per_edge=5000,
                         sigma_range=list(range(25,75,5)),
                         metric = 'explained_variance',
                         plot_folder='./plots/'):
    metric_vals, num_clusters, missing_clusters = [], [], []
    for sigma in sigma_range:
        cluster_out = clustering(data_embedding, filename=None, 
                                 bins_per_edge=bins_per_edge, sigma=sigma)
        num_clusters += [np.max(cluster_out[2])+1]
        missing_clusters += [np.max(cluster_out[2])+1 - len(np.unique(cluster_out[2]))]

        if metric == 'aic':
            curr_metric = get_aic(data_embedding, cluster_out[2])
            print("AIC: ", curr_metric)
            if not metric_vals or all(curr_metric<val for val in metric_vals):
                best_cluster_out = cluster_out
                min_metric_sigma = sigma

            metric_vals += [curr_metric]
        elif metric == 'explained_variance':
            curr_metric = explained_variance(data_embedding, cluster_out[2])
            print("Explained Variance: ", curr_metric)
            metric_vals += [curr_metric]
            

    
    # f = plt.figure()
    # ax = f.add_subplot(111)
    # ax.imshow(best_cluster_out[0])
    # ax.set_aspect('auto')
    # ax.plot(best_cluster_out[1][:,0],best_cluster_out[1][:,1],'.r',markersize=0.05)
    # plt.savefig(''.join([plot_folder,'_watershed_sig',str(min_aic_sigma),'.png']),dpi=400)
    # plt.close()
        
    f = plt.figure()
    plt.plot(sigma_range, metric_vals, marker='o', c='k')
    plt.savefig(''.join([plot_folder,metric,'_curve.png']), dpi=400)
    plt.close()

    f = plt.figure()
    plt.plot(sigma_range, num_clusters, marker='o', c='k')
    plt.savefig(''.join([plot_folder,'num_clusters.png']), dpi=400)
    plt.close()

    f = plt.figure()
    plt.plot(sigma_range, missing_clusters, marker='o', c='k')
    plt.savefig(''.join([plot_folder,'missing_clusters.png']), dpi=400)
    plt.close()

    return 0; #best_cluster_out, min_aic_sigma



def explained_variance(data, data_by_cluster):
    total_variance = np.sum(np.std(data, axis=0)**2)
    k = np.max(data_by_cluster)+1 # Number clusters
    centroids = np.zeros((k,2))
    cluster_n = np.zeros(k) # 
    global_mean = np.mean(data, axis=0)

    for cluster in np.unique(data_by_cluster):
        cluster_data = data[data_by_cluster==cluster,:]
        cluster_n[cluster] = np.shape(cluster_data)[0]
        cluster_mean = np.mean(data[data_by_cluster==cluster,:], axis=0)
        centroids[cluster] = cluster_mean

    # import pdb; pdb.set_trace()
    explained_var = np.sum(cluster_n*np.sum((centroids-global_mean)**2,axis=1)/(k-1))/total_variance
    return explained_var

def get_aic(data, data_by_cluster):
    '''
    IN: 
        data - Embedded coordinates
        data_by_cluster - Cluster associated with each point in data
    '''
    centroids = np.zeros((np.shape(data_by_cluster)[0],2))
    cluster_stds = np.zeros((np.shape(data_by_cluster)[0],2))
    for cluster in np.unique(data_by_cluster):
        # import pdb; pdb.set_trace()
        cluster_std = np.std(data[data_by_cluster==cluster,:], axis=0)
        cluster_std = np.where(cluster_std==0, 1e-5, cluster_std)

        cluster_mean = np.mean(data[data_by_cluster==cluster,:], axis=0)
        centroids[data_by_cluster==cluster] = cluster_mean
        cluster_stds[data_by_cluster==cluster] = cluster_std
    # import pdb; pdb.set_trace()
    sse = np.sum((data-centroids)**2/cluster_stds)/(np.max(data_by_cluster)+1)
    aic = 2*(np.max(data_by_cluster)+1)*60 + np.log(sse)
    return aic

# def paired_ttest(cluster_frequencies, comparison, conditionIDs, animalIDs):
#     for group in comparison:
#         group_freq = cluster_frequencies[]
