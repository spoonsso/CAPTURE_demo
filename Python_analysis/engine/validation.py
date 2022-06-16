import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from embed import Embed, BatchEmbed, Watershed
import matplotlib.pyplot as plt

import tqdm
from typing import Union, Optional, List, Tuple
import time

class KFoldEmbed:
    def __init__(self,
                 k_split: int = 10, 
                 out_path: str = './plots/'):
        self.k_split = k_split
        self.out_path = out_path

        self.mse = []
        self.euc = []

    def run(self,
            embedder: Union[BatchEmbed, Embed], 
            param: str,
            param_range: List[Union[int,float]]):
        '''
        param can be either k, n_tree, or 
        '''
        self.embedder = embedder
        self.param = param
        self.param_range = param_range

        print("Embedding 10-fold data")
        template = embedder.template
        temp_embedding = embedder.temp_embedding
        print("Template shape: ", template.shape)
        print("Predictions shape: ", temp_embedding.shape)

        ws = Watershed(sigma = 14,
                        max_clip = 1,
                        log_out = True,
                        pad_factor = 0.05)
        cluster_true = ws.fit_predict(temp_embedding)
        self.n_clusters = len(np.unique(cluster_true))

        kf = KFold(n_splits = self.k_split, shuffle = True)
        #max distance calculated corner to corner
        max_dist = np.sqrt(np.sum((np.amax(temp_embedding,axis=0)-np.amin(temp_embedding,axis=0))**2))
        metric_vals, min_metric_embedding = [], []
        self.euc, self.mse, self.cluster_acc = [], [], []
        for param_val in param_range:
            setattr(embedder, param, param_val) # seting new param
            print("Reembedding with",param_val, param)
            kf_embedding = np.zeros(np.shape(temp_embedding))
            start = time.time()
            mse_k, euc_k = np.zeros((template.shape[0],1)), np.zeros((template.shape[0],1))

            for train, test in tqdm.tqdm(kf.split(template, temp_embedding)):
                # Embed the 90
                import pdb; pdb.set_trace()
                kf_temp_embedding = embedder.embed(features = template[train],
                                                   save_self = False)

                # Reembed the 10 using the 90
                kf_embedding[test] = embedder.predict(data = template[test],
                                                      template = template[train],
                                                      temp_embedding = kf_temp_embedding)
                
                # kf_embedding = np.append(kf_embedding,
                #                          reembedding,
                #                          axis=0)
            print("Total Time K-Fold Reembedding: ", time.time()-start)

            mse_by_frame = np.sum((temp_embedding - kf_embedding)**2, axis=0) #squared error for each point
            self.mse += [np.mean(mse_by_frame)/(max_dist**2)] # mean squared error/(max distance^2)
            self.euc += [np.mean(np.sqrt(mse_by_frame))/max_dist] # mean euclidean distance/max distance

            # if metric is empty or curr_metric is lowest so far
            if not self.mse:
                self.min_mse_embed = kf_embedding
                self.min_mse_param = param_val
            elif all(self.mse[-1]<val for val in self.mse[:-1]):
                Print("Found new min mse: ", self.mse[-1], " at param ", param_val)
                self.min_mse_embed = kf_embedding
                self.min_mse_param = param_val

            if not self.euc:
                self.min_euc_embed = kf_embedding
                self.min_euc_param = param_val
            elif all(self.euc[-1]<val for val in self.euc[:-1]):
                Print("Found new min Euclidean error: ", self.euc[-1], " at param ", param_val)
                self.min_euc_embed = kf_embedding
                self.min_euc_param = param_val

            cluster_k = ws.predict(kf_embedding)
            self.cluster_acc += [np.sum(cluster_k == cluster_true)]

        print("MSE: ", self.mse)
        print("Euclidean Error: ", self.euc)

        return self

    def plot_error(self,
                   out_path: Optional[str] = None):
        if out_path is None: out_path = self.out_path
        err_dict = {'MSE': self.mse, 
                    'Euclidean': self.euc, 
                    'Cluster_acc': self.cluster_acc}
        
        for key in err_dict.keys():
            f = plt.figure()
            plt.plot(self.param_range, err_dict[key], marker='o', c='k')
            plt.savefig(''.join([out_path,'k_fold_',key,'.png']), dpi=400)
            plt.close()

     