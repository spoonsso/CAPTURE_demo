from DataStruct import DataStruct
from visualization import skeleton_vid3D

# poseD = psd.PoseData(config_path='../embedding_analysis_ws_60.yaml')

# import pdb; pdb.set_trace()

# poseD.load_connectivity()

ds = DataStruct(config_path='../embedding_analysis_ws_r01.yaml')
ds.load_feats(downsample=10)
ds.load_meta()
ds.load_connectivity()
ds.load_preds()

# skeleton_vid3D(ds.pose_3d,
#                ds.connectivity,
#                frames = [3000,100000,2000000],
#                SAVE_ROOT = ''.join([ds.out_path,'/videos/']))

import pdb; pdb.set_trace()