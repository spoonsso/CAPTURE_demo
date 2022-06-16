import pandas as pd
import numpy as np
import scipy.io
import h5py
import hdf5storage
import matplotlib.pyplot as plt
from typing import Optional
from MetaWrapper import MetaWrapper
# import cv2
import time
# import plotly.express as px
# import plotly.graph_objs as go
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# from google.colab.patches import cv2_imshow
from typing import List, Tuple

class FeatureData(MetaWrapper):
    '''
    Rows are frames w/good tracking and after downsampling
    '''
    #TODO: clean these up along with the config param inputs to read config
    _props = [
        'analysis_path',
        'preds_path',
        'meta_path',
        'out_path',
        'embedding_method',
        'exp_key',
        'upsampled',
        'embed_vals',
        'data',
        'meta_by_frame',
        'frame_id',
        'features',
        'meta',
        'downsample',
    ]

    _data_keys=['embed_vals','frame_id','features']

    def __init__(self, 
                 config_path: str,):
        '''
        Initializationss
        upsampled: upsampled rate (here 3)
        vid_length: default 3600s (1hr)
        '''
        config_params = ['analysis_path','preds_path','meta_path','out_path',
                         'results_path','embedding_method','exp_key',
                         'upsampled']
        super().__init__(config_path = config_path, 
                         config_params = config_params)

        self.downsample=None

    @property
    def embed_vals(self):
        return self.data['embed_vals'].to_numpy()

    @embed_vals.setter
    def embed_vals(self,
                   embed_vals: Optional[np.array]=None):
        self.data['embed_vals'] = list(embed_vals)

    @property
    def features(self):
        return np.array(list(self.data['features'].to_numpy())) # seems slow
        # return self.data[list(range(self.feat_shape[1]))].to_numpy()

    @features.setter
    def features(self,
                 features: np.array):

        # self.data.drop([list(range(self.feat_shape[1]))],axis=1)
        # self.data.loc[:,list(range(features.shape[1]))] = features
        self.data['features'] = list(features) # seems slow

    @property
    def feat_shape(self):
        return np.shape(self.features)

    def load_feats(self,
                   analysis_path: Optional[str]=None, 
                   preds_path: Optional[str]=None, 
                   exp_key: Optional[str]=None, 
                   downsample: int = 20, 
                   return_out: bool = False):
        '''
        Load in data (we only care about exp_id, frames_with_good_tracking and jt_features)

        IN:
            analysis_path - Path to MATLAB analysis struct with jt_features included
            preds_path - Path to predictions .mat file
            exp_key - Name of category to separate by experiment
            downsample - Factor by which to downsample features and IDs for analysis

        OUT:
            features - Numpy array of features for each frame for analysis (frames x features)
            exp_id - List of labels for categories based on the exp_key
            frames_with_good_tracking - Indices in merged predictions file to keep track of downsampling
        '''
        if analysis_path: self.analysis_path = analysis_path
        if preds_path: self.preds_path = preds_path
        if exp_key: self.exp_key = exp_key
        self.downsample = downsample

        # import pdb; pdb.set_trace()
        analysisstruct = hdf5storage.loadmat(self.analysis_path, variable_names=['jt_features','frames_with_good_tracking'])
        features = analysisstruct['jt_features']

        try:
            frames_with_good_tracking = np.squeeze(analysisstruct['frames_with_good_tracking'][0][0].astype(int))-1
        except:
            frames_with_good_tracking = np.squeeze(analysisstruct['frames_with_good_tracking'][0][1].astype(int))-1

        exp_ids_full = np.squeeze(hdf5storage.loadmat(self.preds_path, variable_names=[self.exp_key])[self.exp_key].astype(int))

        exp_id = exp_ids_full[frames_with_good_tracking] # Indexing out batch IDs

        print("Size of dataset: ", np.shape(features))


        # downsample
        frames_with_good_tracking = frames_with_good_tracking[::self.downsample]
        features = features[::self.downsample]
        exp_id = exp_id[::self.downsample]

        self.exp_id = exp_id
        self.frame_id = frames_with_good_tracking
        self.features = features

        if return_out:
            return features, exp_id, frames_with_good_tracking

        return self


