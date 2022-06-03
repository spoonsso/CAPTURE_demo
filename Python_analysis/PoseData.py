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

class PoseData(MetaWrapper):

    def __init__(self,
                 config_path: str):
        config_params = ['preds_path','meta_path','out_path','skeleton_path',
                         'exp_key','skeleton_name']
        super().__init__(config_path = config_path, 
                         config_params = config_params)

        # Rows are all frames
        self.pose_3d = None

        self.connectivity = connectivity([],[],[])

    def load_preds(self, 
                   preds_path: Optional[str] = None,
                   connectivity = None,
                   return_out: bool = False):

        if preds_path: self.preds_path = preds_path
        if connectivity: self.connectivity = connectivity

        exp_ids_full = np.squeeze(hdf5storage.loadmat(self.preds_path, variable_names=[self.exp_key])[self.exp_key].astype(int))

        self.exp_id = exp_ids_full
        
        f = h5py.File(preds_path)['predictions']
        total_frames = max(np.shape(f[list(f.keys())[0]]))

        pose_3d = np.empty((total_frames, 0, 3))
        for key in connectivity.joint_names:
            print(key)
            try:
                joint_preds = np.expand_dims(np.array(f[key]).T,axis=1)
            except:
                print("Could not find ",key," in preds")
                continue

            pose_3d = np.append(pose_3d, joint_preds, axis=1)
        
        
        self.pose_3d = pose_3d
        if return_out:
            return pose_3d

        return self

    @property
    def pose_3d(self):
        return self.data['pose_3d']

    @pose_3d.setter
    def pose_3d(self,
                pose_3d: np.ndarray):
        self.data['pose_3d'] = pose_3d

    def load_connectivity(self, 
                          skeleton_path=None, 
                          skeleton_name='mouse20',
                          return_out=False):

        '''
        Load in joint names, connectivities and colors for connectivites
        IN:
            skeleton_path: Path to Python file with dicts for skeleton information
            skeleton_name: Key for correct skeleton in connectivity dict

        '''
        if skeleton_path:
            self.skeleton_path=skeleton_path

        if self.skeleton_path.endswith('.py'):
            import importlib.util
            mod_spec = importlib.util.spec_from_file_location('connectivity',self.skeleton_path)
            con = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(con)

            joints = con.JOINT_NAME_DICT[skeleton_name] # joint names
            colors = con.COLOR_DICT[skeleton_name] # color to be plotted for each linkage
            links = con.CONNECTIVITY_DICT[skeleton_name] # joint linkages

            self.connectivity = connectivity(joints, colors, links)

            if return_out:
                return self.connectivity

        return self

    
class connectivity:

    def __init__(self, 
                 joint_names: List[str], 
                 colors: List[Tuple[float,float,float,float]], 
                 links: List[Tuple[int,int]]):

        self.joint_names=joint_names

        conn_dict = {'links':links,
                     'colors':colors}

        self.conn_df = pd.DataFrame(data=conn_dict)

    @property
    def links(self):
        return self.conn_df['links']

    @links.setter
    def links(self,
              links: List[Tuple[int,int]]):
        self.conn_df['links'] = links

    @property
    def colors(self):
        return self.conn_df['colors']

    @colors.setter
    def colors(self,
               colors: List[Tuple[float,float,float,float]]):
        self.conn_df['colors'] = colors
