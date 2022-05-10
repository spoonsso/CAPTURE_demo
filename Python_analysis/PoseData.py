import pandas as pd
import numpy as np
import scipy.io
import hdf5storage
import matplotlib.pyplot as plt
import cv2
import time
import plotly.express as px
import plotly.graph_objs as go
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from google.colab.patches import cv2_imshow

class AnalysisStruct:

    def __init__(self, myanalysisstruct_dict, predictions_dict, left_or_right_colormap_dict, common_path, upsampled=3, vid_length=3600):
        '''
        Initialization
        common_path: /content
        upsampled: upsampled rate (here 3)
        vid_length: default 3600s (1hr)
        '''
        # read from .mat as dict
        self.myanalysisstruct_dict = myanalysisstruct_dict
        self.predictions_dict = predictions_dict
        self.left_or_right_colormap_dict = left_or_right_colormap_dict

        # parameters
        self.upsampled = upsampled # 2268000 x 3 = 6804000
        self.tsnegranularity = self.myanalysisstruct_dict['tsnegranularity'] # 50 here
        self.nVids = len(self.predictions_dict['fullpath'][0]) # Number of videos
        self.vid_length = vid_length # 3600 secs per video
        print('Number of videos:', self.nVids)
        
        # save paths
        self.common_path = common_path
        self.videopaths = {}
        for i in range(1, self.nVids + 1):
            self.videopaths[i] = self.path_format_switch(self.predictions_dict['fullpath'][0][i-1][0])+'/Camera1/0.mp4'

        # get cluster information (which animalIDs are selected in good tracking)
        self.df_frames_with_good_tracking = pd.DataFrame.from_dict(self.myanalysisstruct_dict['frames_with_good_tracking'][0][0]) - 1 # get from .mat as df. range[1, 6803953] - 1 -> [0, 6803952]
        self.df_zValues = pd.DataFrame.from_dict(self.myanalysisstruct_dict['zValues']).rename(columns={0:'x', 1:'y'}) # get zValues from .mat
        self.df_animalID = pd.DataFrame.from_dict(self.predictions_dict['animalID']) # get animalID from .mat as df
        self.df_expanded_animalID = pd.DataFrame(np.repeat(self.df_animalID.values, \
                                                        self.upsampled, axis=0)) # 2.26M x 3 = 6.80M. idx range[0, 6803999], val range[1,1,1,...,7,7,7]
        self.df_selected_animalID = self.df_expanded_animalID.iloc[self.df_frames_with_good_tracking.values.T.tolist()[0], :].reset_index(drop=True) # get animalIDs selected in good tracking. 134453 x 1
        # append animalID information into zValues
        self.df_tSNE = self.df_zValues.copy()
        self.df_tSNE.insert(2, 'animalID', self.df_selected_animalID) # now columns become [x, y, animalID]

        # get anchor idxs for the 3D anchor plot
        self.selected_anchor_idxs = (self.df_frames_with_good_tracking.values//3).T.tolist()[0] # 3D anchors with good tracking (scale: 2M)
        # append anchor information into zValues
        self.df_tSNE['idx'] = self.selected_anchor_idxs
        self.df_tSNE.reindex(columns=['idx', 'x', 'y', 'animalID'])

    def load_dannce_data(self, ):
    
    def draw_tSNE(self, df_tSNE, color=None, marker_size=3):
        '''
        Draw a 2d tSNE plot from zValues.

        input: zValues dataframe, [num of points x 2]
        output: a scatter plot
        '''
        plt.figure(figsize=[12,10])
        unique_animalID = np.unique(df_tSNE['animalID'])
        for lbl in unique_animalID:
            plt.scatter(x=df_tSNE['x'][df_tSNE['animalID'] == lbl], \
                        y=df_tSNE['y'][df_tSNE['animalID'] == lbl], c=color, label=lbl, s=marker_size)
        plt.legend()
        plt.xlabel('tSNE1')
        plt.ylabel('tSNE2')
        plt.show()

    def draw_tSNE_interactive(self, df_tSNE, color='animalID', marker_size=3):
        '''
        Draw an interactive 2d tSNE plot from zValues.

        input: zValues dataframe, [num of points x 2]
        output: a scatter plot
        '''
        unique_animalID = np.unique(df_tSNE['animalID'])
        fig = px.scatter(df_tSNE, x='x', y='y', color=color, hover_data=['idx', 'x', 'y'], width=800, height=800)
        fig.update_traces(marker_size=marker_size)
        fig.show()

    def draw_3d_skeleton(self, selected_anchor_idx):
        '''
        input: an index of predictions 3d coordinates
        output: a skeleton scatter3d plot of that index
        '''
        selected_anchor_idx = int(selected_anchor_idx)
        temp_list = []
        anchors = list(predictions_dict['predictions'][0][0])[:-1] # 'Tail_base_' etc. excluding last matrix 'sampleID'
        for i in range(len(anchors)): 
            temp_list.append(anchors[i][selected_anchor_idx])

        plt.figure()
        ax = plt.axes(projection="3d")

        for x, y, z in temp_list:
            ax.scatter3D(x, y, z)

        skeleton_color, colors = None, self.left_or_right_colormap_dict['color'].tolist()
        for i, (first, second) in enumerate(self.left_or_right_colormap_dict['joints_idx']):
            xx = [anchors[first-1][selected_anchor_idx][0], anchors[second-1][selected_anchor_idx][0]]
            yy = [anchors[first-1][selected_anchor_idx][1], anchors[second-1][selected_anchor_idx][1]]
            zz = [anchors[first-1][selected_anchor_idx][2], anchors[second-1][selected_anchor_idx][2]]
            if colors[i] == [1, 0, 0]:
                skeleton_color = 'r' 
            elif colors[i] == [0, 1, 0]:
                skeleton_color = 'g'
            else:
                skeleton_color = 'b'
        ax.plot(xx, yy, zz, c=skeleton_color)
        plt.show()

    def draw_3d_skeleton_interactive(self, selected_anchor_idx, marker_size=3):
        '''
        input: an index of predictions 3d coordinates
        output: a skeleton scatter3d plot of that index
        '''
        selected_anchor_idx = int(selected_anchor_idx)
        temp_list = []
        anchors = list(predictions_dict['predictions'][0][0])[:-1] # 'Tail_base_' etc. excluding last matrix 'sampleID'
        for i in range(len(anchors)): 
            temp_list.append(anchors[i][selected_anchor_idx])
        df_temp = pd.DataFrame(temp_list)
        print(df_temp.head())
        fig = px.scatter_3d(temp_list, x=0, y=1, z=2)
        skeleton_color, colors = None, self.left_or_right_colormap_dict['color'].tolist()
        fig = go.Figure()
        for i, (first, second) in enumerate(self.left_or_right_colormap_dict['joints_idx']):
            xx = [anchors[first-1][selected_anchor_idx][0], anchors[second-1][selected_anchor_idx][0]]
            yy = [anchors[first-1][selected_anchor_idx][1], anchors[second-1][selected_anchor_idx][1]]
            zz = [anchors[first-1][selected_anchor_idx][2], anchors[second-1][selected_anchor_idx][2]]
            name = self.left_or_right_colormap_dict['joint_names'][0][first-1].tolist()[0]+'-'\
                    + self.left_or_right_colormap_dict['joint_names'][0][second-1].tolist()[0]
            if colors[i] == [1, 0, 0]:
                skeleton_color = 'r' 
            elif colors[i] == [0, 1, 0]:
                skeleton_color = 'g'
            else:
                skeleton_color = 'b'
            fig.add_scatter3d(x=xx, y=yy, z=zz, name=name)
        fig.update_traces(marker_size=marker_size)
        fig.show()

    def path_format_switch(self, original_path):
        '''
        Switches the path from format
        '/media/twd/dannce-pd/PDBmirror/2021-07-04-PDb1_0-dopa'
        to
        '/hpc/group/tdunn/pdb_data/videos/2021_04_07/PDb1_R1_0/videos/'
        '''
        year, day, month, pdb, _ = original_path.split('/')[5].split('-')
        pdb1, pdb2 = pdb[3], pdb[5]
        converted_path = self.common_path + '/videos/{}_{}_{}/PDb{}_R1_{}/videos'.format(year, month, day, pdb1, pdb2)
        return converted_path

    def get_relative_frame_idx(self, overall_idx):
        '''
        Get the index in a certain video from the overall index. 

        input: Overall index.
        output: frame_number: relative index in a video. v
                id_idx: which video is the frame in. 
                amount_of_frames: amount of frames per video (assuming each video has same length)
        '''
        overall_idx = int(overall_idx)
        example_video = cv2.VideoCapture(self.videopaths[1])
        amount_of_frames = int(example_video.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Frames per video:', amount_of_frames)
        frame_number = overall_idx % amount_of_frames
        vid_idx = overall_idx // amount_of_frames + 1 # find which vid the frame belongs to. E.g., 3 // 324000 + 1 = 0 + 1 = 1. So frame 3 is in 1st video.
        print('The requested frame idx {} is in video number {}, at the {}th frame of that video.'.format(overall_idx, vid_idx, frame_number))
        return frame_number, vid_idx, amount_of_frames

    def frame2time(self, overall_idx):
        '''
        Convert the overall index of a frame (a point in tSNE) to the time stamp (in second).

        input: overall_idx, the index of the frame in all videos (0-2M)
        output: timestamp of the frame in seconds.
        '''
        frame_number, vid_idx, amount_of_frames = self.get_relative_frame_idx(overall_idx)
        return frame_number / amount_of_frames * self.vid_length # the length of vid is 3600s.

    def get_frame(self, overall_idx):
        '''
        Get the frame in the set of videos with a given index.
        For example, there are 7 videos, each with 324000 frames,
        so there are 324000x7=2268000 frames. You can give
        overall_idx = 2000001

        input: overall_idx, the index of the frame in all videos (0-2M)
        output: an .jpg image of that frame
        '''
        print('Starts extracting frame...')
        frame_number, vid_idx, amount_of_frames = self.get_relative_frame_idx(overall_idx)
        video = cv2.VideoCapture(self.videopaths[vid_idx])
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        is_success, frame = video.read()
        frame = cv2.resize(frame, (500, 500))
        cv2_imshow(frame)
        if is_success:
            image_name = "video{}_frame{}.jpg".format(vid_idx, frame_number)
            cv2.imwrite(image_name, frame)
            print('Frame successfully extracted as', image_name)

    def generate_video(self, overall_idx, video_length=3000):
        '''
        Generate a video around the given frame of length video_length (in ms).

        input: overall frame index，desired video length (e.g., 3000ms)
        output: a video fraction
        '''
        print('Starts generating video...')
        frame_number, vid_idx, amount_of_frames = self.get_relative_frame_idx(overall_idx)
        timestamp = self.frame2time(overall_idx)
        half_length = video_length / 2000 # in seconds
        start, end = max(timestamp - half_length, 0), min(timestamp + half_length, self.vid_length)
        video_name = "video{}_time{}m{}s.mp4".format(vid_idx, round(timestamp // 60, 2), round(timestamp % 60, 2))
        ffmpeg_extract_subclip(anst.videopaths[vid_idx], start, end, targetname=video_name)
        print('Video clip {} generated successfully with length of {} secs.'.format(video_name, round(end - start, 2)))

