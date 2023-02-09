function mocapstruct = compute_preprocessed_mocapstruct_demo(mocapstruct_in,preprocessing_parameters)
mocapstruct = mocapstruct_in;
markers = mocapstruct_in.markers_preproc;
markers_preproc = mocapstruct_in.markers_preproc;
fps = mocapstruct_in.fps;

marker_names = fieldnames(markers);
marker_frame_length = size(markers.(marker_names{1}),1);
%  markers_preproc = markers;
%  markers_preproc_aligned = markers_aligned;

%% median filter the data to remove spikes
fprintf('median filtering %f \n')
for ll = 1:numel(marker_names)
    markers_preproc.(marker_names{ll}) = medfilt2(markers_preproc.(marker_names{ll}),[preprocessing_parameters.median_filt_length,1]);
end


%% Transform the data into relevant features for clustering and visualization

%'big-data' features
% get relative marker positions to one another (x,y,z)
num_markers = numel(markers);
marker_velocity = zeros(num_markers,marker_frame_length,4);
marker_position = zeros(num_markers,marker_frame_length,3);
abs_velocity_antialiased = zeros(num_markers,marker_frame_length);


dH = designfilt('lowpassiir', 'FilterOrder', 3, 'HalfPowerFrequency', 60/(300/2), ...
    'DesignMethod', 'butter');
[f1,f2] = tf(dH);

%delta_markers_reshaped = [];
fprintf('getting velocities \n')
for ll = 1:numel(marker_names)
    marker_position(ll,:,1:3) = markers_preproc.(marker_names{ll});
    for mk = 1:3
    end
    marker_velocity(ll,2:(end),1:3) = diff(markers_preproc.(marker_names{ll}),1,1);
    marker_velocity(ll,1,1:3) = marker_velocity(ll,2,1:3);
    marker_velocity(ll,:,4) = sqrt(sum((squeeze( marker_velocity(ll,:,1:3))).^2,2));
    
    % a simple fix to pre-remove bad frames -- depending on the exact
    % quality of your data you may want to change things here. 
    frames_to_mask = union(find(isnan(marker_velocity(ll,:,4))),find(isinf(marker_velocity(ll,:,4))));
    marker_velocity(ll,frames_to_mask,4) = 0;
    abs_velocity_antialiased(ll,:) =  filtfilt(f1,f2, marker_velocity(ll,:,4));
    marker_velocity(ll,frames_to_mask,4) = nan;
    abs_velocity_antialiased(ll,frames_to_mask) = nan;
    
    for lk = (ll+1):num_markers
        distance_here =   (markers_preproc.(marker_names{ll})-markers_preproc.(marker_names{lk}));
    end
end


%get aggregate feature matrix
%% simple bad frame detector
if numel(fieldnames(markers))>3
    
    fprintf('finding bad frames \n')
    bad_frames_agg = getbadframes(marker_velocity,marker_position,fps,preprocessing_parameters);
    clear marker_position
    
    %% Run the imputation and redo the bad frame detection. THis imputation has been replaces, so this just aligns the spine
    [~,markers_preproc_aligned,mean_position,rotation_matrix] = align_hands_elbows(mocapstruct.markers_preproc,fps);
    mocapstruct.markers_aligned_preproc = markers_preproc_aligned;
    mocapstruct.aligned_rotation_matrix = rotation_matrix;
    mocapstruct.aligned_mean_position = mean_position;
    %                        mocapstruct = assign_modular_annotation_properties(mocapstruct,2);
    
    %% run the swap detection/correction
    
    %% run the imputation
    
    %% get rest/move
    %% get move/not move with a simple threshold -- this is improved below
    veltrace = (conv(abs_velocity_antialiased(5,:),ones(1,fps)./fps,'same'));
    vel_thresh = preprocessing_parameters.moving_threshold;
    
    % frames_move_old = find(veltrace>vel_thresh);
    % frames_rest_old = find(veltrace<=vel_thresh);
    
    vel_thresh_fast = preprocessing_parameters.fastvelocity_threshold;
    
    
    %% use new criteria for moving fast and slow . Essential move frames are those within 600 frames of a threshold crossing but the same threshold is used
    [frames_move,frames_rest,veltrace,movethreshparams] = find_moving_frames(markers_preproc,preprocessing_parameters);
    movethreshparams.vel_thresh_fast = vel_thresh_fast;
    
    % frames_move_fast = find(veltrace>vel_thresh_fast);
    % frames_rest_fast = find(veltrace<=vel_thresh_fast);
    
    % fprintf('Old frames moving %f frames resting %f \n',numel(frames_move_old),numel(frames_rest_old));
    fprintf('NEW Frames moving %f frames resting %f frames move fast \n',numel(frames_move),numel(frames_rest));%,numel(frames_move_fast));
    
    
    %             move_frames = zeros(1,numel(veltrace));
    %             move_frames(veltrace>vel_thresh) = 1;
    %             move_frames = conv(move_frames,ones(1,300)./300,'same');
    
    %frames_near_move = find(move_frames >0);
    %  frames_near_rest = setxor(1:numel(veltrace),frames_move);
    
    
    %% move/not move -- can change
    mocapstruct.move_frames = frames_move;
    mocapstruct.rest_frames = frames_rest;
    % mocapstruct.move_frames_fast = frames_move_fast;
    % mocapstruct.rest_frames_fast = frames_rest_fast;
    mocapstruct.bad_frames_agg = bad_frames_agg;
    %% fill the struct fields
    mocap_struct.veltrace = veltrace;
    mocap_struct.movethreshparams = movethreshparams;
    
end