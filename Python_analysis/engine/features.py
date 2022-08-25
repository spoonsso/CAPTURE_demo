import DataStruct as ds
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation
import numpy as np
import visualization as vis
import tqdm

def rotate_floor(pose,
                 exp_id):

    # for i in np.unique(exp_id):

    return 0


def norm_rotate(pose):
    pose_center = pose - np.expand_dims(pose[:,4,:],axis=1)
    
    x_axis = np.repeat(np.expand_dims([1,0,],axis=0), pose_center.shape[0],axis=0)
    # x_projection = np.einsum('xy,xy->x',pose_center[:,3,:2],x_axis)
    # norm = np.sqrt(np.einsum('xy,xy->x',pose_center[:,3,:2],pose_center[:,3,:2])) #magnitudes
    # theta_abs = np.arccos(x_projection/norm)
    # theta = np.where(pose_center[:,3,1]>0, -theta_abs, theta_abs)


    theta = -np.arctan2(pose_center[:,3,1],pose_center[:,3,0])#np.cross(pose_center[:,3,:2],x_axis)/x_projection)
    # theta = np.where(pose_center[:,3,1]>0 & pose_center[:,3,0]<0, theta-np.pi, theta)
    # theta = np.where(pose_center[:,3,1]<0 ^ pose_center[:,3,0]<0, np.pi+theta, theta)
    # theta = -np.arctan(np.cross(pose_center[:,3,:2],x_axis)/x_projection)
    # import pdb; pdb.set_trace()

    # pose_rot = np.zeros(pose_center.shape)
    # for i in tqdm.tqdm(range(len(theta))):
    #     r = Rotation.from_rotvec(theta[i]*np.array([0,0,1]))
    #     # rot_mat = np.array([[np.cos(theta[i]),-np.sin(theta[i]),0],
    #     #                     [np.sin(theta[i]),np.cos(theta[i]),0],
    #     #                     [0,0,1]])
    #     # r = Rotation.from_quat([np.cos(theta[i]/2),0,0,np.sin(theta[i]/2)])
    #     pose_rot[i,:,:] = r.apply(pose_center[i,:,:])ta
    #     # import pdb; pdb.set_trace()
    #     # pose_rot[i,:,:] = np.matmul(rot_mat,pose_center[i,:,:].T).T
    #     # import pdb; pdb.set_trace()

    rot_mat = np.array([[np.cos(theta), -np.sin(theta), np.zeros(len(theta))],
                        [np.sin(theta), np.cos(theta), np.zeros(len(theta))],
                        [np.zeros(len(theta)), np.zeros(len(theta)), np.ones(len(theta))]]).repeat(18,axis=2)#.transpose((2,0,1))
    pose_rot = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose_center,(-1,3))).reshape(pose_center.shape)
    # import pdb; pdb.set_trace()
    pose_rot = median_filter(pose_rot,(5,1,1)) # Median filter
    # import pdb; pdb.set_trace()

    # pose_rot = rot_mat[:1000,:,:] @ pose_center.reshape(-1,3)[:1000,:].T
    # pose_rot1 = np.matmul(rot_mat,np.reshape(pose_center,(pose_center.shape[0]*pose_center.shape[1],3)).T)

    # rot = Rotation.align_vectors(pose_center[:,3,:], x_axis)

    ## Median filter
    return pose_rot




pose_struct = ds.DataStruct(config_path = '../configs/path_configs/embedding_analysis_ws_r01.yaml')
pose_struct.load_connectivity()
pose_struct.load_pose()
pose_struct.load_feats()
import pdb; pdb.set_trace()

pose_rot = norm_rotate(pose_struct.pose_3d)

vis.skeleton_vid3D(pose_rot,
                   pose_struct.connectivity,
                   frames=[1000],
                   N_FRAMES = 300,
                   VID_NAME='rotate_vid.mp4',
                   SAVE_ROOT='./')