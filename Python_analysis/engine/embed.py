import numpy as np
import time
import sys
import math
import os
import visualization as vis
from FeatureData import FeatureData
from typing import Optional, Union, List
import faiss

class BatchTSNE:
    def __init__(self,
                 sampling_n = 20,
                 n_iter = 1000,
                 n_neighbors = 300,
                 perplexity = 'default',
                 lr = 'default',
                 method = 'tsne_cuda',
                 save = Optional[str] = None):

        self.features = data
        self.sampling_n = sampling_n
        self.n = np.shape(self.features)[0]

        if perplexity=='default':
            self.perplexity = max(int(self.n/100),30)
        else:
            self.perplexity=perplexity

        self.n_neighbors = 300
        self.method = method

        if lr = 'default':
            self.lr = int(self.n/12)
        else:
            self.lr = lr

        self.template_ = None
        self.template_idx_ = None


    def fit_transform(self,
                      data: Union[np.array, FeatureData],
                      batchID: Optional[np.array, List[Union[int,str]]] = None,
                      save_batchmaps: Optional[str] = None):

        if self.method=='tsne_cuda':
            import tsnecuda as tc
            for batch in np.unique(batchID):
                data_by_ID = data[batchID == batch,:]
                tsne = tc.TSNE(n_iter=self.n_iter, 
                               verbose=2, 
                               num_neighbors=300, 
                               perplexity=self.perplexity, 
                               learning_rate=self.lr)
                embedding = tsne.fit_transform(data_by_ID)

                if self.save_batchmaps:
                    embed_scatter(embedding, filename=save_batchmaps)

                watershed_map, _, data_by_cluster, _ = clustering(embedding, filename=''.join([filename, str(batch)]), sigma=15)

            sampled_points, idx = sample_clusters(data_by_ID, data_by_cluster, sample_size=self.sampling_n)

            idx = np.nonzero(batch_ID==batch)[0][idx]
            template = np.append(template, sampled_points, axis=0)
            template_idx += list(idx)

        return template

    def sample_clusters(data, 
                        data_by_cluster: Union[np.array, List[int]], 
                        sample_size: int = 20):
        '''
        Equally sampling points from each cluster for a batchmap
        IN:
            data - All of the data in dataset (may be downsampled)
            data_by_cluster - Cluster number for each point in `data`
            size - Number of points to sample from a cluster
        OUT:
            sampled_points - Values of sampled points from `data`
            idx - Index in `data` of sampled points
        '''
        data = np.append(data,np.expand_dims(np.arange(np.shape(data)[0]),axis=1),axis=1)
        sampled_points = np.empty((0,np.shape(data)[1]))
        for cluster_id in np.unique(data_by_cluster):
            points = data[data_by_cluster==cluster_id,:]
            if len(points)==0:
                continue
            elif len(points)<size:
                continue
                # sampled_idx = np.random.choice(np.arange(len(points)), size=size, replace=True)
                # sampled_points = np.append(sampled_points, points[sampled_idx,:], axis=0)
            else:
                num_points = min(len(points),size)
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