function [] = findEmbeddings_20220206(fileName,savePath)
addpath(genpath('./utilities/'));
addpath(genpath('./tSNE/'));

load('/n/holylfs02/LABS/olveczky_lab/Ugne/featureEinfo.mat','mcz','y','parameters','mu','sig');

[a b] = fileparts(fileName);
fullfile = b;
trainingData = mcz;
load(fileName,'ccz');

fprintf(1,'Finding Embeddings\n');
    [zValues,zCosts,zGuesses,inConvHull,meanMax,exitFlags] = ...
        findTDistributedProjections_fmin(ccz,mcz,...
        y,[],parameters);
 
save([savePath '/' fullfile '_RE.mat'],'zValues','zCosts','zGuesses','inConvHull','fileName');
