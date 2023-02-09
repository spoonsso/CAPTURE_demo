%% Demo for Tadross lab behavioral analysis
% ---------------------------
% (C) Jesse D Marshall 2020
%     Harvard University

addpath(genpath('/hpc/group/tdunn/joshwu/CAPTURE_demo/'))
flag = int32(bitor(2,8))
py.sys.setdlopenflags(flag);

predsfile='merged_predictions.mat'

if exist('preprocess_struct','var')==0
    preprocess_struct = 'ratception_struct'
else
    preprocess_struct
end

if exist('tsne_type','var')==0
    tsne_type='gpu'
else
    tsne_type
end

if exist('overwrite_coefficient', 'var')==0
    overwrite_coefficient = 0
else
    overwrite_coefficient
end

if exist('analysis_tag','var')==0
    analysis_tag = ""
else
    analysis_tag
end
    
if exist('coeff_file', 'var')==0
    coeff_file = 'my_coefficients'
else
    coeff_file
end

warning('off','MATLAB:chckxy:IgnoreNaN')

input_params.fps = 90; %what is the fps of the video we are analyzing?
%internal: check dependencies
[fList,pList] = matlab.codetools.requiredFilesAndProducts('pd_analysis_demo.m');

%% run the analysis
basedirectory = '/hpc/group/tdunn/st3dio/analysis/PDb/';
savedirectory = '/hpc/group/tdunn/joshwu/CAPTURE_demo/Species_specific_files/combined_tadross_full/fixed/';
%input predictions in DANNCE format
animfilename = strcat(basedirectory,filesep,predsfile);
%outputfile
animfilename_out = strcat(savedirectory,filesep,preprocess_struct,'.mat');

% input_params.SpineM_marker = 'centerBack';
% input_params.SpineF_marker = 'backHead';
% input_params.conversion_factor = 525; %mm/selman
input_params.repfactor = 1;%floor(300/my_fps);

%% preprocess the data
ratception_struct = preprocess_dannce(animfilename,animfilename_out,'taddy_mouse',input_params);

%% copy over camera information and metadata
predictionsfile = load(animfilename);
if isfield(predictionsfile,'cameras')
    predictionsfieldnames = fieldnames(predictionsfile);
    for lk=1:numel(predictionsfieldnames)
        ratception_struct.(predictionsfieldnames{lk}) = predictionsfile.(predictionsfieldnames{lk});
    end
end
save(animfilename_out,'-struct','ratception_struct','-v7.3')

%%

ratception_struct.predictions = ratception_struct.markers_preproc;
ratception_struct.sample_factor = 1;%floor(300/my_fps);
ratception_struct.shift = 0;

clear ratception_struct;

%% do embedding
[analysisstruct,hierarchystruct] = CAPTURE_quickdemo(...
    animfilename_out,...
    'taddy_mouse',coeff_file,'taddy_mouse',overwrite_coefficient,tsne_type);

disp("saving myanalysisstruct")
save(strcat(savedirectory,filesep,'myanalysisstruct',analysis_tag,'.mat'),'-struct','analysisstruct',...
    '-v7.3')
% save(strcat(basedirectory,filesep,'myhierarchystruct.mat'),'-struct','hierarchystruct',...
%     '-v7.3')

%% plot the tsne
plotfolder = strcat(savedirectory,filesep,'plots/');
mkdir(plotfolder)

h1=figure(608)
clf;
params.nameplot=0;
params.density_plot =1;
params.watershed = 1;
params.sorted = 0;
params.markersize = 0.2;
params.jitter = 0;
params.coarseboundary = 0;
analysisstruct.params.density_width=0.25;
analysisstruct.params.density_res=4001;
plot_clustercolored_tsne(analysisstruct,1,params.watershed,h1,params)
set(gcf,'renderer','painters')
%colorbar off
axis equal
set(gcf,'Position',([100 100 1100 1100]))
print('-dpng',strcat(plotfolder,'taddysne.png'),'-r1200')
print('-depsc',strcat(plotfolder,'taddysne.eps'),'-r1200')

disp("Finished!")