import os
import numpy as np
import scipy.io as sio
import imageio
import tqdm
import connectivity
import hdf5storage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

def load_predictions(PRED_EXP = '/home/exx/Desktop/GitHub/CAPTURE_demo/CAPTURE_data/full_tadross_data/merged_predictions.mat', ):
    pred_path = os.path.join(PRED_EXP)
    import h5py
    f = h5py.File(pred_path)['predictions']
    total_frames = max(np.shape(f[list(f.keys())[0]]))
    ANIMAL= 'mouse20'
    JOINTS = connectivity.JOINT_NAME_DICT[ANIMAL]
    # short_CONNECTIVITY = connectivity.CONNECTIVITY_DICT[ANIMAL]
    # num_joints = max(max(short_CONNECTIVITY))+1

    pose_3d = np.empty((total_frames, 0, 3))
    for key in JOINTS:
        print(key)
        try:
            joint_preds = np.expand_dims(np.array(f[key]).T,axis=1)
        except:
            print("Could not find ",key," in preds")
            continue

        pose_3d = np.append(pose_3d, joint_preds, axis=1)
    return pose_3d

def skeleton_vid3D(preds,
                   frames=[3000,100000,5000000], 
                   VID_NAME = '0.mp4',
                   EXP_ROOT = './initial_tadross_analysis/skeleton_vids/'):
    ###############################################################################################################
    N_FRAMES = 250
    START_FRAME = np.array(frames) - int(N_FRAMES/2) + 1
    ANIMAL= 'mouse20'
    COLOR = connectivity.COLOR_DICT[ANIMAL]*len(frames)
    short_CONNECTIVITY = connectivity.CONNECTIVITY_DICT[ANIMAL]
    CONNECTIVITY = short_CONNECTIVITY
    total_frames = N_FRAMES*len(frames)#max(np.shape(f[list(f.keys())[0]]))
    num_joints = max(max(short_CONNECTIVITY))+1
    for i in range(len(frames)-1):
        next_con = [(x+(i+1)*num_joints, y+(i+1)*num_joints) for x,y in short_CONNECTIVITY]
        CONNECTIVITY=CONNECTIVITY+next_con
    # import pdb; pdb.set_trace()
    JOINTS = connectivity.JOINT_NAME_DICT[ANIMAL]
    SAVE_ROOT = EXP_ROOT #'/media/mynewdrive/datasets/dannce/demo/markerless_mouse_2'

    vid_path = os.path.join(EXP_ROOT, 'videos') 

    save_path = os.path.join(SAVE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get dannce predictions
    pose_3d = np.empty((0, num_joints, 3))
    for start in START_FRAME:
        pose_3d = np.append(pose_3d, preds[start:start+N_FRAMES,:,:],axis=0)

    # compute 3d grid limits 
    offset = 50
    x_lim1, x_lim2 = np.min(pose_3d[:, :, 0])-offset, np.max(pose_3d[:, :, 0])+offset
    y_lim1, y_lim2 = np.min(pose_3d[:, :, 1])-offset, np.max(pose_3d[:, :, 1])+offset
    z_lim1, z_lim2 = np.minimum(0, np.min(pose_3d[:, :, 2])), np.max(pose_3d[:, :, 2])+10
    # open videos
    # vids = [imageio.get_reader(os.path.join(vid_path, cam, VID_NAME)) for cam in CAMERAS]

    # set up video writer
    metadata = dict(title='dannce_visualization', artist='Matplotlib')
    writer = FFMpegWriter(fps=30, metadata=metadata)

    ###############################################################################################################
    # setup figure
    fig = plt.figure(figsize=(12, 12))

    ax_3d = fig.add_subplot(1, 1, 1, projection='3d')

    with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=300):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab imgs
            curr_frames = curr_frame + np.arange(len(frames))*N_FRAMES
            kpts_3d = np.reshape(pose_3d[curr_frames,:,:], (len(frames)*num_joints, 3))

            
            # plot 3d moving skeletons
            ax_3d.scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2],  marker='.', color='black', linewidths=0.5)
            for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
                xs, ys, zs = [np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]]) for j in range(3)] 
                ax_3d.plot3D(xs, ys, zs, c=color, lw=2)

            ax_3d.set_xlim(x_lim1, x_lim2)
            ax_3d.set_ylim(y_lim1, y_lim2)
            ax_3d.set_zlim(z_lim1, z_lim2)
            ax_3d.set_title("3D Tracking")
            # ax_3d.set_box_aspect([1,1,1])

            # grab frame and write to vid
            writer.grab_frame()
            ax_3d.clear()
    
    plt.close()
    return 0