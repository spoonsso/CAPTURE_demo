function ML_features = create_behavioral_features(mocapstruct,coeff_file,overwrite_coeff,linkname)
% Make features for tsne
% ---------------------------
% (C) Jesse D Marshall 2020
%     Harvard University 

%% compute the joint angles11


ML_features = compute_joint_angles_demo(mocapstruct,linkname);
%% compute the principal components of the joint angles and
modular_cluster_properties_clipped_index_8 = mocapstruct.modular_cluster_properties.clipped_index{8};
disp("clearing mocapstruct in create_behavioral_features")

disp("Clearing unnecessary mocapstruct fields for memory")
whos
[status,cmdout] = system('free -h','-echo');
evalin('caller','clear mocapstruct;')
% evalin('caller','mocapstruct = rmfield(mocapstruct, {''aligned_rotation_matrix'',''p_max'', ''predictions'',''x''})')
% % ''aligned_mean_position'', ''markers_preproc'',''markers_aligned_preproc'', 
clear mocapstruct;
ML_features = rmfield(ML_features, {'joint_angles_mean','transverse_seglengths'})

whos
[status,cmdout] = system('free -h','-echo');
ML_features = compute_appendage_pc_demos(modular_cluster_properties_clipped_index_8,ML_features,coeff_file,overwrite_coeff);

disp("Clearing joint angle and segement length features from ML_features for memory")
ML_features = rmfield(ML_features, {'all_seglengths','all_segments'}) %'jointangle_struct'
%% compute the wavelet transform
tic
ML_features = compute_wl_transform_features_demo(ML_features,coeff_file,overwrite_coeff);
toc

disp("Making new ML_features with only necessary features")
%% add window/vel/old features
a = struct();
a.dyadic_spectrograms_score_wl_appendages_euc = ML_features.dyadic_spectrograms_score_wl_appendages_euc;
a.dyadic_spectrograms_score_wl_appendages = ML_features.dyadic_spectrograms_score_wl_appendages;
a.appendage_pca_score_lengths = ML_features.appendage_pca_score_lengths;
a.appendage_pca_score = ML_features.appendage_pca_score;
a.appendage_pca_score_euc = ML_features.appendage_pca_score_euc;
a.frames_appendage_gps = ML_features.frames_appendage_gps;
clear ML_features;
ML_features = a;
% ML_features = struct('dyadic_spectrograms_score_wl_appendages_euc', ML_features.dyadic_spectrograms_score_wl_appendages_euc, ...
%                      'dyadic_spectrograms_score_wl_appendages', ML_features.dyadic_spectrograms_score_wl_appendages, ...
%                      'appendage_pca_score_lengths', ML_features.appendage_pca_score_lengths, ...
%                      'appendage_pca_score', ML_features.appendage_pca_score, ...
%                      'appendage_pca_score_euc', ML_features.appendage_pca_score_euc, ...
%                      'frames_appendage_gps', ML_features.frames_appendage_gps)


%% code for visualization
