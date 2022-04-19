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


def skeleton_vid3D(frames, 
                   PRED_EXP = '/home/exx/Desktop/GitHub/CAPTURE_demo/CAPTURE_data/full_tadross_data/', 
                   EXP_ROOT = './initial_tadross_analysis/skeleton_vids/'):
    ###############################################################################################################
    N_FRAMES = 3000
    VID_NAME = "0.mp4"
    START_FRAME = frames - int(N_FRAMES/2) + 1
    ANIMAL= 'mouse20'
    COLOR = connectivity.COLOR_DICT[ANIMAL]
    CONNECTIVITY = connectivity.CONNECTIVITY_DICT[ANIMAL]
    SAVE_ROOT = EXP_ROOT #'/media/mynewdrive/datasets/dannce/demo/markerless_mouse_2'

    vid_path = os.path.join(EXP_ROOT, 'videos') 
    pred_path = os.path.join(PRED_EXP)

    save_path = os.path.join(SAVE_ROOT, PRED_EXP, 'vis')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ###############################################################################################################
    # load camera parametersnum
    # cameras = load_cameras(os.path.join(EXP_ROOT, LABEL3D_FILE))

    # get dannce predictions
    # pred_3d = sio.loadmat(os.path.join(pred_path, 'merged_predictions.mat'))['pred'][START_FRAME: START_FRAME+N_FRAMES]
    # pred_3d = hdf5storage.loadmat(os.path.join(pred_path, 'merged_predictions.mat'), variable_names=['predictions'])[0]
    import h5py
    f = h5py.File(os.path.join(pred_path, 'merged_predictions.mat'))['predictions']
    print(f)
    total_frames = max(np.shape(f[(list(f.keys())[0], 0)]))
    num_joints = max(max(CONNECTIVITY))

    pose_3d = np.empty((total_frames, 0, 3))
    for key in f.keys():
        if key=='sampleID':
            continue
        else:
            np.unsqueeze(np.array(f[key]).T, 1)


    # get 3d coms
    # com_3d = sio.loadmat(os.path.join(pred_path, 'com3d_used.mat'))['com'][START_FRAME: START_FRAME+N_FRAMES]

    # compute projectionsf
    # pred_2d, com_2d = {}, {}
    # pose_3d = np.transpose(pred_3d, (0, 2, 1)) #[n_samples, n_joints, 3]
    # com_3d = np.expand_dims(com_3d, 1)
    # pts = np.concatenate((pose_3d, com_3d), axis=1)
    # num_chan = pts.shape[1]
    # pts = np.reshape(pts, (-1, 3))
    # for cam in CAMERAS:
    #     projpts = project_to_2d(pts,
    #                             cameras[cam]["K"],
    #                             cameras[cam]["r"],
    #                             cameras[cam]["t"])[:, :2]

    #     projpts = distortPoints(projpts,
    #                             cameras[cam]["K"],
    #                             np.squeeze(cameras[cam]["RDistort"]),
    #                             np.squeeze(cameras[cam]["TDistort"]))
    #     projpts = projpts.T
    #     projpts = np.reshape(projpts, (-1, num_chan, 2))
    #     pred_2d[cam] = projpts[:, :num_chan-1, :]
    #     com_2d[cam] = projpts[:, -1:, :]

    # del projpts

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
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    axes_2d = [ax1, ax2]
    ax_3d = fig.add_subplot(1, 3, 3, projection='3d')

    with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=300):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab imgs
            # imgs = [vid.get_data(curr_frame) for vid in vids]
            kpts_3d = pose_3d[curr_frame]

            # # plot 2d projections
            # for i, cam in enumerate(CAMERAS):
            #     kpts_2d = pred_2d[cam][curr_frame]
            #     com = com_2d[cam][curr_frame]

            #     axes_2d[i].imshow(imgs[i])
            #     axes_2d[i].scatter(kpts_2d[:, 0], kpts_2d[:, 1], marker='.', color='white', linewidths=0.5)
            #     axes_2d[i].scatter(com[:, 0], com[:, 1], marker='.', color='red', linewidths=1)

            #     for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
            #         xs, ys = [np.array([kpts_2d[index_from, j], kpts_2d[index_to, j]]) for j in range(2)]
            #         axes_2d[i].plot(xs, ys, c=color, lw=2)
            #         del xs, ys

            #     axes_2d[i].set_title(cam)
            #     axes_2d[i].axis("off")
            
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
    return 0