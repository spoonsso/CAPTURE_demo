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
import tqdm

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage import measure
import pickle

class BatchTSNE:
    def __init__(self,
                 sampling_n: int = 20,
                 n_iter: int = 1000,
                 n_neighbors: int = 300,
                 perplexity: Union[str, int] = 'auto',
                 lr: Union[str, int] = 'auto',
                 method: str = 'tsne_cuda',
                 sigma: int = 15):

        '''
        t-SNE parameters here are used in the embedding of batches, 
        not for the final template itself
        '''
                 
        self.sampling_n = sampling_n
        self._n = None
        self.n_iter = n_iter

        self._perplexity = perplexity
        self._lr = lr

        self.n_neighbors = n_neighbors
        self.method = method

        self.sigma = sigma

        self.template = None
        self.temp_idx = []
        self.temp_embedding = None

    @property
    def perplexity(self):
        if self._perplexity == 'auto':
            return max(int(self._n/100),30)
        else:
            return self._perplexity

    @perplexity.setter
    def perplexity(self,
                   perplexity: Union[str, int] = 'auto'):
        self._perplexity = perplexity

    @property
    def lr(self):
        if self._lr == 'auto':
            return int(self._n/12)
        else:
            return self._lr

    @lr.setter
    def lr(self,
           lr: Union[str, int] = 'auto'):
        self._lr = lr

    def fit(self,
            data: Union[np.ndarray, DataStruct],
            batch_id: Optional[Union[np.ndarray, List[Union[int,str]]]] = None,
            save_batchmaps: Optional[str] = None,
            save_temp_scatter: Optional[str] = None):
        '''
        '''
        if save_batchmaps:
            save_path = ''.join([save_batchmaps,'/batch_maps/'])
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            filename = ''.join([save_path, self.method])

        if self.method=='tsne_cuda':
            self.template = np.empty((0,data.shape[1]))
            self.template_idx = []

            for batch in tqdm.tqdm(np.unique(batch_id)):
                data_by_ID = data[batch_id == batch,:] # Subsetting data by batch
                self._n = np.shape(data)[0]

                # intializing tsne_cuda
                tsne = tc.TSNE(n_iter=self.n_iter, 
                               verbose=2,
                               num_neighbors=self.n_neighbors,
                               perplexity=self.perplexity,
                               learning_rate=self.lr)
                embedding = tsne.fit_transform(data_by_ID)

                ws = Watershed(sigma=self.sigma,
                               n_bins=1000,
                               max_clip=1,
                               log_out=True,
                               pad_factor=0)
                cluster_labels = ws.fit_predict(embedding)

                if save_batchmaps:
                    ws.plot_density(filepath = ''.join([filename,str(batch),'_density.png']),
                                    watershed = True)
                    vis.scatter(embedding, filepath=''.join([filename,str(batch),'_scatter.png']))

                sampled_points, idx = self.__sample_clusters(data_by_ID, 
                                                            cluster_labels, 
                                                            sample_size=self.sampling_n)
                # import pdb; pdb.set_trace()
                idx = np.nonzero(batch_id==batch)[0][idx]
                self.template = np.append(self.template, sampled_points, axis=0)
                self.temp_idx += list(idx)

        self.embed_template(save_scatter = save_temp_scatter)
        return self

    def embed_template(self,
                       n_iter: Optional[int] = None,
                       n_neighbors: Optional[int] = None,
                       perplexity: Optional[Union[str, int]] = None,
                       lr: Optional[Union[str, int]] = None,
                       method: Optional[str] = None,
                       save_scatter: Optional[str] = None):
        '''
        Calculate t-SNE embedding of template values
        '''
        self._n = self.template.shape[0]
        if not n_iter: n_iter = self.n_iter
        if not n_neighbors: n_neighbors = self.n_neighbors
        if not perplexity: perplexity = self.perplexity
        if not lr: lr = self.lr
        if not method: method = self.method

        if self.method=='tsne_cuda':
            tsne = tc.TSNE(n_iter=self.n_iter, 
                           verbose=2,
                           num_neighbors=self.n_neighbors, 
                           perplexity=self.perplexity, 
                           learning_rate=self.lr)
            self.temp_embedding = tsne.fit_transform(self.template)

            if save_scatter is not None:
                vis.scatter(data=self.temp_embedding, 
                            filepath=''.join([save_scatter,'temp_scatter.png']))
        return self


    def predict(self,
                data: Union[np.ndarray, DataStruct],
                k: int=5):
        '''
        Uses KNN to embed points onto template
        
        IN:
            data - n_frames x n_features
        OUT:
            embed_vals - KNN reembedded values
        '''
        print("Predicting using KNN")
        # from KNNEmbed import KNNEmbed
        start = time.time()
        knn = KNNEmbed(k=k)
        knn.fit(self.template,self.temp_embedding)
        embed_vals = knn.predict(data, weights='distance')
        print("Total Time embedding: ", time.time()-start)

        return embed_vals

    def fit_predict(self,
                    data: Union[np.ndarray, DataStruct],
                    batch_id: Optional[Union[np.ndarray, List[Union[int,str]]]] = None,
                    save_batchmaps: Optional[str] = None, 
                    save_temp_scatter: Optional[str] = None,
                    k: int=5):

        self.fit(data = data,
                 batch_id = batch_id,
                 save_batchmaps = save_batchmaps,
                 save_temp_scatter = save_temp_scatter)
        embed_vals = self.predict(data, k = k)

        return embed_vals

    def __sample_clusters(self,
                          data, 
                          meta_name: Union[np.ndarray, List[Union[int,str]]], 
                          sample_size: int = 20):
        '''
        Equally sampling points from 
        IN:
            data - All of the data in dataset (may be downsampled)
            meta_name - Cluster number for each point in `data`
            sample_size - Number of points to sample from a cluster
        OUT:
            sampled_points - Values of sampled points from `data`
            idx - Index in `data` of sampled points
        '''
        data = np.append(data,np.expand_dims(np.arange(np.shape(data)[0]),axis=1),axis=1)
        sampled_points = np.empty((0,np.shape(data)[1]))
        for meta_id in np.unique(meta_name):
            points = data[meta_name==meta_id,:]
            if len(points)<sample_size:
                # If fewer points, just skip (probably artifactual cluster)
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

    def save_pickle(self,
                    filepath: str = './plot_folder/'):
        pickle.dump(self, open(''.join([filepath,'batch_tsne.p']),"wb"))
        return self

    def load_pickle(self,
                    filepath: str = './plot_folder/batch_tsne.p'):
        self = pickle.load(open(filepath,"rb"))
        return self

class KNNEmbed:
    '''
    Using faiss to run k-Nearest Neighbors algorithm for embedding of points in 2D
    when given high-D data and low-D embedding of template data
    '''
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        '''
        Creates data structure for fast search of neighbors
        IN:
            X - Features of training data
            y - Training data 
        '''
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.ascontiguousarray(X,dtype=np.float32))
        self.y = np.ascontiguousarray(y,dtype=np.float32)

    def predict(self, X, weights='standard'):
        '''
        Predicts embedding of data using KNN
        IN:
            X - Features of data to predict
            weights - 'standard' or 'distance' determines weights on nearest neighbors
        OUT:
            predictions - output predictions
        '''
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
    
class GaussDensity:
    '''
    Class for creating Gaussian density maps of 2D scatter data
    '''
    def __init__(self,
                 sigma: int=15,
                 n_bins: int=1000,
                 max_clip: float=0.75,
                 log_out: bool=False,
                 pad_factor: float=0.025):

        self.sigma = sigma
        self.n_bins = n_bins
        self.max_clip = max_clip
        self.log_out = log_out
        self.pad_factor = pad_factor

        self.hist_range = None

        #TODO: More consideration for when these save
        self.density = None
        self.data_in_bin = None 

    def hist(self,
             data: np.ndarray,
             new: bool=True):
        '''
        Run 2D histogram 
        
        IN:
            data - Data to convert to density map (n_frames x 2)
            new - Map onto old hist range and bins if False
        OUT:
            hist - Calculated 2d histogram  (n_bins x n_bins)
        '''
        range_len = (np.ceil(np.amax(data, axis=0)) - np.floor(np.amin(data, axis=0))).astype(int)
        padding = range_len*self.pad_factor

        # Calculate x and y limits for histogram and density
        if new or (self.hist_range is None):
            print("Calculating new histogram ranges")
            self.hist_range = [[int(np.floor(np.amin(data[:,0]))-padding[0]),int(np.ceil(np.amax(data[:,0]))+padding[0])],
                               [int(np.floor(np.amin(data[:,1]))-padding[1]),int(np.ceil(np.amax(data[:,1]))+padding[1])]]

        hist, self.xedges, self.yedges = np.histogram2d(data[:,0], data[:,1], bins=[self.n_bins, self.n_bins],
                                            range=self.hist_range,
                                            density=False)
        hist = np.rot90(hist)

        assert (self.xedges[0]<self.xedges[-1]) and (self.yedges[0]<self.yedges[1])

        return hist

    def fit_density(self,
                    data: np.ndarray,
                    new: bool=True,
                    map_bin: bool=True):

        '''
        Calculate Gaussian density for 2D embedding

        IN:
            data - Data to convert to density map (n_frames x 2)
            new - Map onto old hist range and bins if False
        OUT:
            density - Calculated density map (n_bins x n_bins)
        '''
        # 2D histogram
        hist = self.hist(data, new)

        # Calculates density using gaussian filter
        density = gaussian_filter(hist, sigma=self.sigma)
        if self.log_out:
            density = np.log1p(density)
        density = np.clip(density, None, np.amax(density)*self.max_clip) # clips max for better visualization of clusters

        if map_bin:
            # Maps each data point to bin indices and saves to self
            # May need some more consideration for when this saves and doesn't save
            self.data_in_bin = self.map_bins(data)

        if new:
            self.density = density

        return density
        
    def map_bins(self,
                 data: np.ndarray):
        '''
        Find which bin in histogram/density map each data point is a part of
        IN:
            edges: self.xedges and self.yedges must be calculated from np.histogram (represents edge values of bins)
            data: Data to be transformed
        OUT:
            data_in_bin: Indices (returns n_frames x 2) of data in density map (shape n_bins x n_bins)
        '''
        if self.xedges is None:
            print("Could not find histogram, computing now")
            self.density = None
            self.hist(data, new=True)

        data_in_bin = np.zeros(np.shape(data))
        for i in range(data_in_bin.shape[0]):
            data_in_bin[i,1] = np.argmax(self.xedges>data[i,0])-1
            data_in_bin[i,0] = self.n_bins-np.argmax(self.yedges>data[i,1])

        return data_in_bin
        
    def plot_density(self,
                     filepath: str = './plot_folder/density.png'):
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.imshow(self.density)
        ax.set_aspect('auto')
        plt.savefig(filepath,dpi=400)
        plt.close()


class Watershed(GaussDensity):

    density_thresh = 1e-5

    def __init__(self,
                 sigma: int=15,
                 n_bins: int=1000,
                 max_clip: float=0.75,
                 log_out: bool=False,
                 pad_factor: float=0.025):

        super().__init__(sigma = sigma,
                         n_bins = n_bins,
                         max_clip = max_clip,
                         log_out = log_out,
                         pad_factor = pad_factor)

        self.watershed_map = None
        self.borders = None

        self.density = None #TODO: Consider more when this saves and doesn't

    def fit(self, 
            data: Union[DataStruct, np.ndarray]):
        '''
        Running watershed clustering on data
        IN:
            data - DataStruct object or numpy array (frames x 2) of t-SNE coordinates
        OUT:
            self.density
        '''
        if isinstance(data, DataStruct):
            data_ = data.embed_vals.values
        else:
            data_ = data

        self.density = self.fit_density(data_,
                                        new=True,
                                        map_bin=False)

        print("Calculating watershed")
        self.watershed_map = watershed(-self.density,
                                        mask=self.density>self.density_thresh,
                                        watershed_line=False)
        self.borders = np.empty((0,2))

        for i in range(1, len(np.unique(self.watershed_map))):
            contour = measure.find_contours(self.watershed_map.T==i, 0.5)[0]
            self.borders = np.append(self.borders, contour,axis=0)

        return self

    def predict(self,
                data: Optional[Union[DataStruct, np.ndarray]]=None):
        '''
            Predicts the cluster label of data

            Requires knowledge of what bin data is in in the histogram/density map

            IN:
                data - XY coordinates of data to be predicted
            OUT:
                cluster_labels - cluster labels of all data
        '''

        data_in_bin = self.map_bins(data)

        cluster_labels = self.watershed_map[data_in_bin[:,0].astype(int),
                                            data_in_bin[:,1].astype(int)]
        print(str(int(np.amax(cluster_labels)+1)),"clusters detected")
        print(str(np.unique(cluster_labels).shape),"unique clusters detected")
        print(np.unique(cluster_labels))

        return cluster_labels

    def fit_predict(self,
                    data: Optional[Union[DataStruct, np.ndarray]]=None):
        self.fit(data)
        cluster_labels = self.predict(data)
        return cluster_labels

    def plot_watershed(self,
                      filepath: str='./plot_folder/watershed.png',
                      borders: bool = True):
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.imshow(self.watershed_map)
        ax.set_aspect('auto')
        if borders:
            ax.plot(self.borders[:,0],self.borders[:,1],'.r',markersize=0.05)
        plt.savefig(''.join([filename,'_watershed.png']),dpi=400)
        plt.close()

    def plot_density(self,
                     filepath: str='./plot_folder/density.png',
                     watershed: bool = True):
        f = plt.figure()
        ax = f.add_subplot(111)
        if watershed:
            ax.plot(self.borders[:,0],self.borders[:,1],'.r',markersize=0.1)
        ax.imshow(self.density)
        ax.set_aspect('auto')
        plt.savefig(filepath,dpi=400)
        plt.close()

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