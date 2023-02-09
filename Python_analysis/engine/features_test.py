from features import *
import DataStruct as ds
import visualization as vis
from sklearn.decomposition import IncrementalPCA

import numpy as np

pose_struct = ds.DataStruct(config_path = '../configs/path_configs/embedding_analysis_dcc_r01.yaml')
pose_struct.load_connectivity()
pose_struct.load_pose()
pose_struct.load_feats()

# Separate videos have rotated floor planes - this rotates them back
pose = align_floor(pose_struct.pose_3d, 
                       pose_struct.exp_ids_full)

# vis.skeleton_vid3D(pose,
#                    pose_struct.connectivity,
#                    frames=[1000],
#                    N_FRAMES = 300,
#                    VID_NAME='vid_rotated.mp4',
#                    SAVE_ROOT='./')

# Calculating velocities and standard deviation of velocites over windows
abs_vel = get_velocities(pose, 
                         pose_struct.exp_ids_full, 
                         pose_struct.connectivity.joint_names)

pose = center_spine(pose)
pose = rotate_spine(pose)
# vis.skeleton_vid3D(pose,
#                    pose_struct.connectivity,
#                    frames=[1000],
#                    N_FRAMES = 300,
#                    VID_NAME='vid_centered.mp4',
#                    SAVE_ROOT='./')

euclid_vec = get_ego_pose(pose,
                          pose_struct.connectivity.joint_names)

angles = get_angles(pose,
                    pose_struct.connectivity.angles)

ang_vel = get_angular_vel(angles,
                          pose_struct.exp_ids_full)

# head_angv = get_head_angular(pose, pose_struct.exp_ids_full)

ipca = IncrementalPCA(n_components=10, batch_size=100)
import pdb; pdb.set_trace()
feat_dict = {
    'euc_vec': euc_vec,
    'angles': angles,
    'abs_vel': abs_vel,
}
import pdb; pdb.set_trace()
pca_feat = {}
for feat in feat_dict:
    pca_feat[feat] = ipca.fit_transform(feat_dict[feat])
# link_lengths = get_lengths(pose,pose_struct.exp_ids_full,pose_struct.connectivity.links)
# feats = np.concatenate((euc_vec, angles, abs_vel, head_angular),axis=1)

import pdb; pdb.set_trace()
#mean center before pca, separate for

# w_let = wavelet(feats_pca)

import pdb; pdb.set_trace()