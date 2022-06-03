import numpy as np
import time
import sys
import math
import os
import visualization as vis
from DataStruct import DataStruct
from typing import Optional, Union, List
import faiss
import tsnecuda as tc

class BatchTSNE:
    def __init__(self,
                 sampling_n: int = 20,
                 n_iter: int = 1000,
                 n_neighbors: int = 300,
                 perplexity: Union[str, int] = 'auto',
                 lr: Union[str, int] = 'auto',
                 method: str = 'tsne_cuda',
                 save: Optional[str] = None):

        self.features = data
        self.sampling_n = sampling_n
        self.n = np.shape(self.features)[0]

        self._perplexity = perplexity
        self._lr = lr

        self.n_neighbors = n_neighbors
        self.method = method

        self.template_ = None
        self.template_idx_ = None

    @property
    def perplexity(self):
        if self._perplexity == 'auto':
            return max(int(self.n/100),30)
        else:
            return self._perplexity

    @perplexity.setter
    def perplexity(self,
                   perplexity: Union[str, int] = 'auto'):
        self._perplexity = perplexity

    @property
    def lr(self):
        if self._lr == 'auto':
            return int(self.n/12)
        else:
            return self._lr

    @lr.setter
    def lr(self,
           lr: Union[str, int] = 'auto'):
        self._lr = lr

    def fit(self,
            data: Union[np.array, DataStruct],
            batchID: Optional[Union[np.array, List[Union[int,str]]]] = None,)


    def fit_transform(self,
                      data: Union[np.array, DataStruct],
                      batchID: Optional[Union[np.array, List[Union[int,str]]]] = None,
                      save_batchmaps: Optional[str] = None):

        if self.method=='tsne_cuda':
            for batch in np.unique(batchID):
                data_by_ID = data[batchID == batch,:]

                # running t-sne cuda
                tsne = tc.TSNE(n_iter=self.n_iter, 
                               verbose=2, 
                               num_neighbors=self.num_neighbors, 
                               perplexity=self.perplexity, 
                               learning_rate=self.lr)
                embedding = tsne.fit_transform(data_by_ID)

                if self.save_batchmaps:
                    vis.scatter(embedding, filename=save_batchmaps)

                _,_,data_by_cluster,_ = clustering(embedding, filename=''.join([filename, str(batch)]), sigma=15)

            sampled_points, idx = self.__sample_clusters(data_by_ID, data_by_cluster, sample_size=self.sampling_n)

            idx = np.nonzero(batch_ID==batch)[0][idx]
            template = np.append(template, sampled_points, axis=0)
            template_idx += list(idx)

        return template


    def sample_clusters(self,
                        data, 
                        meta_name: Union[np.array, List[int]], 
                        sample_size: int = 20):
        '''
        Equally sampling points from 
        IN:
            data - All of the data in dataset (may be downsampled)
            meta_name - Cluster number for each point in `data`
            size - Number of points to sample from a cluster
        OUT:
            sampled_points - Values of sampled points from `data`
            idx - Index in `data` of sampled points
        '''
        data = np.append(data,np.expand_dims(np.arange(np.shape(data)[0]),axis=1),axis=1)
        sampled_points = np.empty((0,np.shape(data)[1]))
        for meta_id in np.unique(meta_name):
            points = data[meta_name==meta_id,:]
            if len(points)<sample_size:
                continue
                # sampled_idx = np.random.choice(np.arange(len(points)), size=size, replace=True)
                # sampled_points = np.append(sampled_points, points[sampled_idx,:], axis=0)
            else:
                num_points = min(len(points),sample_size)
                sampled_points = np.append(sampled_points, 
                                        np.random.permutation(points)[:num_points], 
                                        axis=0)
        print("Number of points sampled")
        print(sampled_points.shape)
        return sampled_points[:,:-1],np.squeeze(sampled_points[:,-1]).astype(int).tolist()


class KNNEmbed:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.ascontiguousarray(X,dtype=np.float32))
        self.y = np.ascontiguousarray(y,dtype=np.float32)

    def predict(self, X, weights='standard'):
        print("Predicting")
        distances, indices = self.index.search(np.ascontiguousarray(X,dtype=np.float32), k=self.k)
        votes = self.y[indices]

        if weights=='distance':
            min_dist = np.min(distances[np.nonzero(distances)])/2
            distances = np.clip(distances, min_dist, None)
            weights = 1/distances
            weights = weights/np.repeat(np.expand_dims(np.sum(weights, axis=1), axis=1), self.k, axis=1)
        else:
            weights = 1/self.k

        weights = np.repeat(np.expand_dims(weights, axis=2), 2, axis=2)
        predictions = np.sum(votes*weights, axis=1)
        return predictions

class Watershed:
    def __init__(self,
                 sigma: int=15,
                 n_bins: int=1000):
        self = self
    


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