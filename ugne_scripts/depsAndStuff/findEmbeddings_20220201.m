function [] = findEmbeddings_20220201(fileName,savePath)
addpath(genpath('./utilities/'));
addpath(genpath('./tSNE/'));

xIdx = 1:23; yIdx = 1:23;
[X Y] = meshgrid(xIdx,yIdx);
X = X(:); Y = Y(:);
IDX = find(X~=Y);
x1 = 4; x2 = 6; mf = 10; smf = 25;
nx = length(xIdx);


load('/n/holylfs02/LABS/olveczky_lab/Ugne/vecsVals95.mat','vecs','vals','mu');
load('/n/holylfs02/LABS/olveczky_lab/Ugne/trainingData20220201.mat','ydata','cD')
[a b] = fileparts(fileName);
fullfile = b;
trainingData = cD;
load(fileName,'ma1');

nn1 = size(ma1,1);
p1Dist = zeros(nx^2,size(ma1,1));
for i = 1:size(p1Dist,1)
    p1Dist(i,:) = returnDist3d(squeeze(ma1(:,:,X(i))),squeeze(ma1(:,:,Y(i))));
end
p1Dsmooth = zeros(size(p1Dist));
for i = 1:size(p1Dist,1)
    p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
end
p1Dist = p1Dsmooth(IDX,:);
p1z = zeros(nx,nn1);
for i = 1:nx
    p1z(i,:) = smooth(medfilt1(squeeze(ma1(:,3,xIdx(i))),3),3);
end

vecs20 = vecs(:,1:20);
minF = .5; maxF = 20; pcaModes = 20; numModes = pcaModes;
parameters = setRunParameters([]);
parameters.samplingFreq = 50;
parameters.minF = minF;
parameters.maxF = maxF;

p1 = bsxfun(@minus,p1Dist',mu);
proj = p1*vecs20;
[data,~] = findWavelets(proj,numModes,parameters);
n = size(p1Dist,2);
amps = sum(data,2);
data2 = log(data);
data2(data2<-5) = -5;
jv = zeros(n,length(xIdx));
for j = 1:length(xIdx)
    jv(:,j) = [0; medfilt1(sqrt(sum(diff(squeeze(ma1(:,:,xIdx(j)))).^2,2)),30)];
end
jv(jv>=5) = 5;
nnData = [data2 .1*p1z' jv];

fprintf(1,'Finding Embeddings\n');
    [zValues,zCosts,zGuesses,inConvHull,meanMax,exitFlags] = ...
        findTDistributedProjections_fmin(nnData,trainingData,...
        ydata,[],parameters);
 
save([savePath '/' fullfile '_RE.mat'],'zValues','zCosts','zGuesses','inConvHull','fileName');
