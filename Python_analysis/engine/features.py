import DataStruct as ds
import scipy as scp
# from scipy.ndimage import median_filter, convolve
from scipy.spatial.transform import Rotation as R
import numpy as np
import visualization as vis
import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from typing import Optional, Union, List, Tuple

def align_floor(pose: np.array,
                exp_id: Union[List, np.array],
                foot_id: Optional[int] = 12,
                head_id: Optional[int] = 0,
                plot_folder: Optional[str] = None):
    '''
    Due to calibration, predictions may be rotated on different axes
    Rotates floor to same x-y plane per video
    IN:
        pose: 3d matrix of (#frames x #joints x #coords)
        exp_id: Video ids per frame
        foot_id: ID of foot to find floor
    OUT:
        pose_rot: Floor aligned poses (#frames x #joints x #coords)
    '''
    pose_rot = pose
    for i in np.unique(exp_id): # Separately for each video
        pose_exp = pose[exp_id == i,:,:]
        pose_exp = scp.ndimage.median_filter(pose_exp,(5,1,1)) # Median filter 5 frames repeat the ends of video

        # Initial calculation of plane to find outlier values
        [xy,z] = [pose_exp[:,foot_id,:2],pose_exp[:,foot_id,2]]
        const = np.ones((pose_exp.shape[0],1))
        coeff = np.linalg.lstsq(np.append(xy,const,axis=1),z,rcond=None)[0]
        z_diff = (pose_exp[:,foot_id,0]*coeff[0] + pose_exp[:,foot_id,1]*coeff[1] + coeff[2]) - pose_exp[:,foot_id,2]
        z_mean = np.mean(z_diff)
        z_range = np.std(z_diff)*1.5
        mid_foot_vals = np.where((z_diff>z_mean-z_range) & (z_diff<z_mean+z_range))[0] # Removing outlier values of foot

        # Recalculating plane with outlier values removed
        [xy,z] = [pose_exp[mid_foot_vals,foot_id,:2],pose_exp[mid_foot_vals,foot_id,2]]
        const = np.ones((xy.shape[0],1))
        coeff = np.linalg.lstsq(np.append(xy,const,axis=1),z,rcond=None)[0]
        
        # Calculating rotation matrices
        un = np.array([-coeff[0],-coeff[1],1])/np.linalg.norm([-coeff[0],-coeff[1],1])
        vn = np.array([0,0,1])
        theta = np.arccos(np.clip(np.dot(un,vn),-1,1))
        rot_vec = np.cross(un,vn)/np.linalg.norm(np.cross(un,vn))*theta
        rot_mat = R.from_rotvec(rot_vec).as_matrix()
        rot_mat = np.expand_dims(rot_mat, axis=2).repeat(pose_exp.shape[0]*pose_exp.shape[1],axis=2)
        pose_exp[:,:,2]-=coeff[2] # Fixing intercept to zero
        
        # Rotating
        pose_rot[exp_id==i,:,:] = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose_exp,(-1,3))).reshape(pose_exp.shape)

        if plot_folder:
            xx, yy=np.meshgrid(range(-300,300,10),range(-300,300,10))
            zz = coeff[0]*xx + coeff[1]*yy + coeff[2]
            fig = plt.figure(figsize=(20,20))
            ax = plt.axes(projection='3d')
            # ax.scatter3D(pose_exp[1000,foot_id,0], pose_exp[1000,foot_id,1], pose_exp[1000,foot_id,2],s=1000,c='r')
            ax.scatter3D(pose_exp[:,foot_id,0], pose_exp[:,foot_id,1], pose_exp[:,foot_id,2],s=1,)
            ax.plot_surface(xx,yy,zz,alpha=0.2)
            plt.savefig(''.join([plot_folder,'/before_rot',str(i),'.png']))
            plt.close()

            fig = plt.figure(figsize=(20,20))
            ax = plt.axes()
            ax.scatter(pose_rot[exp_id==i,foot_id,0], pose_rot[exp_id==i,foot_id,2],s=1)
            # ax.scatter(pose_rot[1000,foot_id,0], pose_rot[1000,foot_id,2],s=100,c='r')
            plt.savefig('./after_rot',str(i),'.png')
            plt.close()

        ## Checking to make sure snout is on average above the feet
        assert(np.mean(pose_rot[exp_id==i,head_id,2])>np.mean(pose_rot[exp_id==i,foot_id,2])) #checking head is above foot

    return pose_rot


def spine_center_rot(pose):
    '''
    Centers mid spine to (0,0,0) and aligns spine_m -> spine_f to x-z plane
    IN:
        pose: 3d matrix of (#frames x #joints x #coords)
    OUT:
        pose_rot: Centered and rotated pose (#frames x #joints x #coords)
    '''
    pose_center = pose - np.expand_dims(pose[:,4,:],axis=1)
    theta = -np.arctan2(pose_center[:,3,1],pose_center[:,3,0])

    rot_mat = np.array([[np.cos(theta), -np.sin(theta), np.zeros(len(theta))],
                        [np.sin(theta), np.cos(theta), np.zeros(len(theta))],
                        [np.zeros(len(theta)), np.zeros(len(theta)), np.ones(len(theta))]]).repeat(18,axis=2)#.transpose((2,0,1))
    pose_rot = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose_center,(-1,3))).reshape(pose_center.shape)

    return pose_rot

def get_angles(pose,
               link_pairs):
    '''
    Calculates 3 angles for pairs of linkage vectors
    Angles calculated are those between projections of each vector onto the 3 xyz planes
    IN:
        pose: Centered and rotated pose (#frames, #joints, #)
    '''
    angles = np.zeros((pose.shape[0],len(link_pairs),3))
    for i,pair in enumerate(link_pairs):
        v1 = pose[:,pair[0],:]-pose[:,pair[1],:]
        v2 = pose[:,pair[2],:]-pose[:,pair[1],:]
        
        angles[:,i,0] = np.arctan2(v1[:,0],v1[:,1]) - np.arctan2(v2[:,0],v2[:,1])
        angles[:,i,1] = np.arctan2(v1[:,0],v1[:,2]) - np.arctan2(v2[:,0],v2[:,2])
        angles[:,i,2] = np.arctan2(v1[:,1],v1[:,2]) - np.arctan2(v2[:,1],v2[:,2])
    
    # Fix all negative angles so that final is between 0 and 2pi
    angles = np.where(angles>0, angles, angles+2*np.pi)
    # import pdb; pdb.set_trace()
    return angles

def get_lengths(pose,
                linkages):
    # import pdb; pdb.set_trace()
    linkages = np.array(linkages)
    lengths = np.square(pose[:,linkages[:,1],:]-pose[:,linkages[:,0],:])
    lengths = np.sum(np.sqrt(lengths),axis=2)
    return lengths

def get_velocities(pose,
                   exp_id,
                   joints=[0,3,4,5],
                   widths=[10,100,300]):

    vel_feats = np.zeros((pose.shape[0],len(joints)*len(widths)))
    for i in np.unique(exp_id):
        pose_exp = pose[exp_id==i,joints,:]
        # Calculate distance beetween  times t - (t-1)
        temp_pose = np.append(np.expand_dims(pose_exp[0,:,:],axis=0),pose_exp[:-1,:,:],axis=0)
        dv = np.sqrt(np.sum(np.square(pose_exp-temp_pose),axis=2))
        # Calculate average velocity over the windows
        for j,width in enumerate(widths):
            kernel = np.ones((width,1))/width
            vel_feats[exp_id==i,j*len(joints):(j+1)*len(joints)] = scp.ndimage.convolve(dv, kernel, mode='constant')

        # import pdb; pdb.set_trace()

    return vel_feats

def get_head_angular(pose,
                     exp_id,
                     widths=[10,100,300],
                     link = [0,3,4]):
    '''
    Getting x-y angular velocity of head

    '''
    v1 = pose[:,0,:2]-pose[:,3,:2]
    v2 = pose[:,4,:2]-pose[:,3,:2]

    angle= np.arctan2(v1[:,0],v1[:,1]) - np.arctan2(v2[:,0],v2[:,1])
    angle = np.where(angle>0, angle, angle+2*np.pi)

    angular_vel = np.zeros((len(angle),len(widths)))
    for i in np.unique(exp_id):
        angle_exp = angle[exp_id==i]
        d_angv = angle_exp - np.append(angle_exp[0],angle_exp[:-1])
        for i,width in enumerate(widths):
            kernel = np.ones(width)/width
            angular_vel[exp_id==i,i] = scp.ndimage.convolve(d_angv, kernel, mode='constant')

    return angular_vel

def wavelet(features,
            sample_freq = 90,
            freq = np.linspace(1,25),
            w0 = 5,):
    # scp.signal.morlet2(500, )
    widths = w0*sample_freq/(2*freq*np.pi)
    for i in range(features.shape[1]):
        freq_transform = scp.signal.cwt(features[:,i],scp.signal.morlet2, widths, w=w0)
    


    return cwtm

pose_struct = ds.DataStruct(config_path = '../configs/path_configs/embedding_analysis_dcc_r01.yaml')
pose_struct.load_connectivity()
pose_struct.load_pose()
pose_struct.load_feats()

pose_rot = align_floor(pose_struct.pose_3d, pose_struct.exp_ids_full, conn=pose_struct.csonnectivity)
velocities = get_velocities(pose_rot, pose_struct.exp_ids_full)

pose_rot = spine_center_rot(pose_rot)
euc_vec = np.reshape(pose_rot, (pose_rot.shape[0],pose_rot.shape[1]*pose_rot.shape[2]))

angles = get_angles(pose_rot,pose_struct.connectivity.angles)
angles = np.reshape(angles,(angles.shape[0],angles.shape[1]*angles.shape[2]))

link_lengths = get_lengths(pose_rot,pose_struct.exp_ids_full,pose_struct.connectivity.links)
head_angular = get_head_angular(pose_rot, pose_struct.exp_ids_full)
feats = np.concatenate((euc_vec, angles, link_lengths, velocities, head_angular),axis=1)

import pdb; pdb.set_trace()
#mean center before pca, separate for 
ipca = IncrementalPCA(n_components=30, batch_size=100)
feats_pca = ipca.fit_transform(feats)

w_let = wavelet(feats_pca)

import pdb; pdb.set_trace()

vis.skeleton_vid3D(pose_rot,
                   pose_struct.connectivity,
                   frames=[1000],
                   N_FRAMES = 300,
                   VID_NAME='rotate_vid.mp4',
                   SAVE_ROOT='./')