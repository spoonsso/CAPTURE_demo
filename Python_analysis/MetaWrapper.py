from dataclasses import dataclass
import pandas as pd
import numpy as np
import scipy.io
import h5py
import hdf5storage
import matplotlib.pyplot as plt
from typing import Optional, Union, List

class MetaWrapper():

    def __init__(self,
                 data: pd.DataFrame = pd.DataFrame(),
                 meta: pd.DataFrame = pd.DataFrame(),
                 config_path: Optional[str] = None,
                 config_params: Optional[List[str]] = None):
        self.config_path = config_path
        # Paths and params from config

        (
            self.analysis_path,
            self.preds_path,
            self.meta_path,
            self.out_path,
            self.results_path,
            self.skeleton_path,
            self.embedding_method,
            self.exp_key,
            self.upsampled,
            self.skeleton_name
        ) = tuple(self.read_config(config_path).values())

        self.data = data
        self.meta = meta

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx=tuple(idx)

        self.data = self.data.loc[idx]

        return self
        
    def check_reset_data(self, len:int):
        if self.data.shape[0] != len:
            self.data = pd.DataFrame()
    
    def read_config(self,
                    filepath,
                    config_params: Optional[List[str]] = None):
        '''
        Read configuration file and set instance attributes 
        based on key, value pairs in the config file

        IN:
            filepath - Path to configuration file
        OUT:
            config_dict - Dict of path variables to data in config file
        '''
        if config_params is None:
            config_params = ['analysis_path','preds_path','meta_path','out_path',
                        'results_path','skeleton_path','embedding_method','exp_key',
                        'upsampled','skeleton_name']

        import yaml
        with open(filepath) as f:
            config_dict = yaml.safe_load(f)

        for key in config_params:
            if key not in config_dict:
                config_dict[key]=None

        return config_dict
        
    def load_meta(self, 
                  meta_path: Optional[str] = None,
                  exp_id: Optional[List[Union[str, int]]]=None,
                  return_out: bool = False):
        if meta_path: self.meta_path = meta_path
        if exp_id: self.exp_id = exp_id

        meta = pd.read_csv(self.meta_path)
        meta_by_frame = meta.iloc[self.exp_id].reset_index().rename(columns={'index':'exp_id'})

        self.meta = meta
        self.meta_by_frame = meta_by_frame

        if return_out:
            return meta, meta_by_frame

        return self

    def fix_vid_paths(self,):
        # if self.meta['Base Directory'].values[0]:
        return self

    @property
    def frame_id(self):
        return self.data['frame_id']

    @frame_id.setter
    def frame_id(self,
                 frame_id: Union[List[int],np.array]):
        self.data['frame_id'] = frame_id
    
    @property
    def exp_id(self):
        return self.data['exp_id']

    @exp_id.setter
    def exp_id(self,
               exp_id: Union[List[Union[str, int]],np.array]):
        self.data['exp_id'] = exp_id

    @property
    def meta_by_frame(self):
        return self.data[self.meta.columns.values.tolist()]

    @meta_by_frame.setter
    def meta_by_frame(self,
                      meta_by_frame: pd.DataFrame):
        self.data.loc[:, meta_by_frame.columns.values.tolist()] = meta_by_frame.values

    def meta_unique(self,
                    column_id: str):
        return list(set(self.data[column_id].values.to_list))

    def n_frames(self):
        return self.data.shape[0]

    def write_pickle(self,
                     out_path: Optional[str] = None):
        import pickle
        pickle.dump(self,out_path)