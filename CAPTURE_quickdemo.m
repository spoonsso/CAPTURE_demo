


function [analysisstruct,hierarchystruct] =  CAPTURE_quickdemo(inputfile,ratnames,coefficientfilename,linkname,overwrite_coefficient,tsne_type,savedirectory)
% File to generate tsne features and run reembedding on a mouse
%      inputfile: a .mat file that contains a preprocessed dannce struct
%                 (see preprocess_dannce)
%      ratnames: a string containing the name of the experiment/rat to be
%                used in saving files
%      coefficientnames: preexisting names of coefficients or file to save
%                      tsne coefficients to. 
%      linkname: the name of the animal (ie kyle_mouse or 'rats' or 'bird'
%                 to be used in computing tsne features)
%      
% This repository contains contributions from the following FEX/Open source contributions which are included:
%Chronux
%Pca Randomized
%MTimesX
%othercolor
%Motionmapper
%Structvars
% ---------------------------
% (C) Jesse D Marshall 2020
%     Harvard University 



%load mocap file
if isempty(inputfile)
datafile = ...
    load('/hpc/group/tdunn/lq53/cap3/CAPTURE_demo/Species_specific_files/predictions.mat');
else
    datafile = load(inputfile);
    if isstruct(datafile) && numel(fieldnames(datafile)) == 1
        fname = fieldnames(datafile);
        datafile = datafile.(fname{1});
    end
end
mocapstruct = datafile;
clear datafile;


if isempty(coefficientfilename)
    coefficient_file = 'demo_coefficients.mat';
else
    coefficient_file = coefficientfilename;
end

if isempty(ratnames)
    ratname = 'myrat';
else
    ratname = ratnames;
end


if isempty(ratnames)
    ratname = 'myrat';
else
    ratname = ratnames;
end

% could change
savefilename ='myextratsnefeature';
directory_here = pwd;

%feature filename and whether or not to overwrite
MLmatobjfile = 'myMLfeatures.mat';
overwrite_MLmatobjfile = 1;

%visualize the mocap data
%animate_markers_nonaligned_fullmovie_demo(mocapstruct,1:10:10000);

%% Create behavioral features
%this determines the set of frames to use -- in general if the animal is
%resting for too long it will cause errors
mocapstruct.modular_cluster_properties.clipped_index{8} = 1:size(mocapstruct.aligned_mean_position,1 );

% to control the wavelet parameters, you can change the properties in the
% compute_wl_transform_features file
if ~exist(MLmatobjfile,'file') || overwrite_MLmatobjfile
MLmatobj = create_behavioral_features(mocapstruct,coefficient_file,overwrite_coefficient,linkname);
else
    MLmatobj = load(MLmatobjfile);
end

if isempty(inputfile)
    datafile = ...
        load('/hpc/group/tdunn/lq53/cap3/CAPTURE_demo/Species_specific_files/predictions.mat');
    else
        datafile = load(inputfile);
        if isstruct(datafile) && numel(fieldnames(datafile)) == 1
            fname = fieldnames(datafile);
            datafile = datafile.(fname{1});
        end
    end
mocapstruct = datafile;
clear datafile;
% perform a tsne embedding subselecting every 50 frames
analysisparams.tsnegranularity = 1;

%subselect a particular set of features
analysisstruct = compute_tsne_features(MLmatobj,mocapstruct,analysisparams);
aligned_mean_position = mocapstruct.aligned_mean_position;
clear mocapstruct
clear MLmatobj
%run tsne
disp("whos post jtfeatures")
whos
[status,cmdout] = system('free -h','-echo');
% savedirectory = '/hpc/group/tdunn/joshwu/CAPTURE_demo/Species_specific_files/combined_tadross_full/fixed/';
save(strcat(savedirectory,filesep,'anstruct_restinc_bonus_feat.mat'),'-struct','analysisstruct',...
    '-v7.3')
disp(size(analysisstruct.jt_features))
if tsne_type=='old'
    %%% Old tsne
    tic
    zvals = tsne(analysisstruct.jt_features);
    toc
elseif tsne_type=='gpu'
    %%% GPU tSNE
    pyenv;
    np = py.importlib.import_module("numpy");
    py.importlib.import_module("tsne_gpu");
    features_np = py.numpy.array(analysisstruct.jt_features(:).');
    features_np = features_np.reshape(py.int(size(analysisstruct.jt_features,1)), py.int(size(analysisstruct.jt_features,2)));
    % tic
    zvals = double(py.tsne_gpu.tsne_gpu(features_np));
    % toc
else
    zvals = 0
end
zvals=0

% save('tsne_embeddings.mat','zvals','zvals_gpu')
% zvals_gpu_size = size(zvals_gpu)
% pause

figure(1)
plot(zvals(:,1),zvals(:,2),'ob','MarkerFaceColor','b')
analysisstruct.zValues = zvals;

%% clustering parameters
analysisstruct.params.density_res = 1001; %resolution of the map
analysisstruct.params.density_width = 2; %density kernel in tsne space
analysisstruct.params.expansion_factor = 1.1; %add a little room to the map after kernel smoothing
analysisstruct.params.density_threshold = 1*10^(-5); %remove regions in plots with low density
analysisstruct.matchedconds = {[unique(analysisstruct.condition_inds)]}; %if running over multiple conditions
analysisstruct.conditions_to_run = [unique(analysisstruct.condition_inds)];
analysisstruct.tsnegranularity = analysisparams.tsnegranularity;
params.reorder=1;
% analysisstruct = compute_analysis_clusters_demo(analysisstruct,params);

%% behavior plots and movies
analysisstruct.conditionnames = ratname;
analysisstruct.ratnames = ratname;
analysisstruct.filesizes = {size(aligned_mean_position,1 );};

%% plot a tsne map -- see plotting script for parameter definitions
h1=figure(609)
clf;
params.nameplot=0;
params.density_plot =0;
params.watershed = 1;
params.sorted = 1;
params.markersize = 1;
params.coarseboundary =0;
params.do_coarse = 0;
% plot_clustercolored_tsne(analysisstruct,1,params.watershed,h1,params)
set(gcf,'Position',([100 100 1100 1100]))


% bird specific axes
axisparams.zlim = ([200 300]);
axisparams.xlim = ([-400 400]);
axisparams.ylim = ([-400 400]);
figure(370);
clf;
% animate_markers_nonaligned_fullmovie_demo(analysisstruct.mocapstruct_reduced_agg{1},...
%     find(analysisstruct.annot_reordered{end}==10),[],axisparams);
% animate_markers_aligned_fullmovie_demo(analysisstruct.mocapstruct_reduced_agg{1},...
%     find(analysisstruct.annot_reordered{end}==59));

%% or use extnded set of 140 features
if strcmp(ratname,'myrat')
MLmatobj_extra = create_extra_behavioral_features(mocapstruct,'myrat',savefilename,overwrite_coefficient,directory_here);
jt_features_extra = load_extra_tsne_features(mocapstruct,MLmatobj_extra,analysisparams);

save(strcat(savedirectory,filesep,'jt_features_extra.mat'),'-struct','jt_features_extra',...
    '-v7.3')

% look at tsne of these added features
zvals_extra = tsne(jt_features_extra);
% or the combination
%zvals_extra = tsne(cat(2,analysisstruct.jt_features,jt_features_extra));
figure(2)
plot(zvals_extra(:,1),zvals_extra(:,2),'ob','MarkerFaceColor','b')
analysisstruct.zValues_extra = zvals;

end

%% run sequence and state analysis
params.do_show_pdistmatrix =1;
params.decimation_factor = 5; %downsample if needed to save on memory
params.doclustering = 1;

%clustering parameters
params.corr_threshold = 0.2;
params.clustercutoff = 0.65;
analysisstruct.plotdirectory = '';
%timescale to use, in seconds
params.timescales = [1./4 2]; 

analysisstruct.conditionnames = {'test'};
analysisstruct.ratname = {ratname};
hierarchystruct = 0;
% hierarchystruct=   find_sequences_states_demo(analysisstruct,1,params);

%animate_markers_nonaligned_fullmovie_demo(analysisstruct.mocapstruct_reduced_agg{1},...
%    find(hierarchystruct.clustered_behavior{1}==2));

end
%visualize


%internal: check dependencies
%[fList,pList] = matlab.codetools.requiredFilesAndProducts('CAPTURE_quickdemo.m');
