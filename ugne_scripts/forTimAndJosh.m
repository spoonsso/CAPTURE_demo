%% loading ART/BUD/COLT
% HERES WHERE I FOUND ALL MY FILENAMES
% list all files w/ dannce

cd /Volumes/olveczky_lab_holy2/Everyone/dannce_rig/dannce_ephys/art/
d = dir;

for i = 1:size(d,1)
artF{i} = d(i).name;
end
artF = artF(3:end-1);


cd /Volumes/olveczky_lab_holy2/Everyone/dannce_rig/dannce_ephys/bud/
d = dir;

for i = 1:size(d,1)
budF{i} = d(i).name;
end
budF = budF(3:end);



cd /Volumes/olveczky_lab_holy2/Everyone/dannce_rig/dannce_ephys/coltrane/
d = dir;

for i = 1:size(d,1)
coltF{i} = d(i).name;
end
coltF = coltF(3:end);


% SAVED FOR EASY RELOADING
save('EphysBEH/foldernames.mat','artF','budF','coltF')



%% LOADING EACH FILE
% SMOOTHING THE DANNCE PREDICTIONS, MAKING ROTATED/ALIGNED DATA
artJ = cell(size(artF)); 
budJ = cell(size(budF));
coltJ = cell(size(coltF));
artJz = cell(size(artF)); 
budJz = cell(size(budF));
coltJz = cell(size(coltF));
for i = 1:length(artF)
    i
    %fname = ['/Volumes/olveczky_lab_holy2/Everyone/dannce_rig/dannce_ephys/art/' artF{i} '/DANNCE/predict03/save_data_AVG.mat'];
    fname = ['/run/user/1000/gvfs/smb-share:server=olveczky.rc.fas.harvard.edu,share=olveczky_lab_holy2/EVeryone/dannce_rig/dannce_ephys/art/' artF{i} '/DANNCE/predict03/save_data_AVG.mat'];
    
    artF2{i} = fname;
    try
        load(artF2{i});
        for j = 1:23
            for k = 1:3
                pred(:,k,j) = smooth(medfilt1(pred(:,k,j),5),5);
            end
        end
        [t,tzs] = alignDannceNF(pred);
         artJ{i} = t;
         artJz{i} = pred;
    catch
        fprintf(['Didnt load number ' num2str(i) '\n']);
    end
end

for i = 1:length(budF)
    i
    %fname = ['/Volumes/olveczky_lab_holy2/Everyone/dannce_rig/dannce_ephys/bud/' budF{i} '/DANNCE/predict02/save_data_AVG.mat'];
    fname = ['/run/user/1000/gvfs/smb-share:server=olveczky.rc.fas.harvard.edu,share=olveczky_lab_holy2/EVeryone/dannce_rig/dannce_ephys/bud/' budF{i} '/DANNCE/predict02/save_data_AVG.mat'];
    
    budF2{i} = fname;
    try
        load(budF2{i});
        for j = 1:23
            for k = 1:3
                pred(:,k,j) = smooth(medfilt1(pred(:,k,j),5),5);
            end
        end
        [t,tzs] = alignDannceNF(pred);
         budJ{i} = t;
         budJz{i} = pred;
    catch
    end
end


for i = 1:length(coltF)
    i
    % fname = ['/Volumes/olveczky_lab_holy2/Everyone/dannce_rig/dannce_ephys/coltrane/' coltF{i} '/DANNCE/predict00/save_data_AVG.mat'];
    fname = ['/run/user/1000/gvfs/smb-share:server=olveczky.rc.fas.harvard.edu,share=olveczky_lab_holy2/EVeryone/dannce_rig/dannce_ephys/coltrane/' coltF{i} '/DANNCE/predict00/save_data_AVG.mat'];
    
    coltF2{i} = fname;
    try
        load(coltF2{i});
        for j = 1:23
            for k = 1:3
                pred(:,k,j) = smooth(medfilt1(pred(:,k,j),5),5);
            end
        end
        [t,tzs] = alignDannceNF(pred);
         coltJ{i} = t;
         coltJz{i} = pred;
    catch
    end
end

ratid = [ones(42,1); 2*ones(28,1); 3*ones(25,1)];
fx1 = zeros(size(artJ));
for i = 1:length(fx1)
    fx1(i) = ~isempty(artJ{i});
end

allJJ = [artJ(fx1==1) budJ coltJ];
allJZ = [artJz(fx1==1) budJz coltJz];
allNames = [artF2(fx1==1) budF2 coltF2];

% allRot = cell(size(allJ));
% for i = 1:length(allRot)
%     temp = allJS{i};
%     [t,~] = alignDannceNF(temp);
%     allRot{i} = t;
% end


% LOADING THE SKELETON
skeleton = load('/home/ugne/Dropbox/ugneDANNCE/rat23.mat');
skeleton = load('/Users/ugne/Dropbox/ugneDANNCE/rat23.mat')
xIdx = 1:23; yIdx = 1:23;
[X Y] = meshgrid(xIdx,yIdx);
X = X(:); Y = Y(:);
IDX = find(X~=Y);
x1 = 4; x2 = 6; mf = 10; smf = 25; links = skeleton.joints_idx;

% HERE I WROTE A FUNCTION TO FIND DISTANCES IN 3D AND SAVED EACH DIST MATRIX
% FOR ONLINE PCA
nx = length(xIdx);
mdist = cell(size(allJS)); mz = cell(size(allJS));
for f = 1:95
    
    f
    ma1 = allJS{f};
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
    save(['dist95/info_' num2str(f) '.mat'],'p1Dist','p1z','-v7.3');
    clear p1Dist p1Dsmooth p1z
end


% online PCA
% OK THEN LOAD EACH DIST MATRIX AGAIN AND DO THE ONLINE PCA
firstBatch = true;
currentImage = 0;
batchSize = 20000;
for j = 1:241
    fprintf(1,['Processing batch # ' num2str(j) '\n']);
    load(['dist95/info_' num2str(j) '.mat']);
    p1Dist = p1Dist';
    if firstBatch
        firstBatch = false;
        if size(p1Dist,1) < batchSize
            cBatchSize = size(p1Dist,1);
            X = p1Dist;
        else
            cBatchSize = batchSize;
            X = p1Dist;
        end
        currentImage = cBatchSize;
        mu = sum(X);
        C = cov(X).*cBatchSize + (mu'*mu)./ cBatchSize;
    else
        if size(p1Dist,1) < batchSize
            cBatchSize = size(p1Dist,1);
            X = p1Dist;
        else
            cBatchSize = batchSize;
            X = p1Dist(randperm(size(p1Dist,1),cBatchSize),:);
        end
        tempMu = sum(X);
        mu = mu + tempMu;
        C = C + cov(X).*cBatchSize + (tempMu'*tempMu)./cBatchSize;
        currentImage = currentImage + cBatchSize;
    end
end

% FIND AND SAVE FINAL MU,VECS,VALS FROM PCA - MIGHT NEED LATER WHEN
% REEMBEDDING IF YOU DONT SAVE INTERMEDIATES
L = currentImage; mu = mu ./ L; C = C ./ L - mu'*mu;
fprintf(1,'Finding Principal Components\n');
[vecs,vals] = eig(C); vals = flipud(diag(vals)); vecs = fliplr(vecs);
save('EphysBEH/vecsVals95.mat','vecs','vals','mu')

% make projections, add z
% IM USING 20 MODES
vecs20 = vecs(:,1:20);
minF = .5; maxF = 20; pcaModes = 20; numModes = pcaModes;
parameters = setRunParameters([]); % THIS IS IN GORDON'S CODE
parameters.samplingFreq = 50; % THIS IS FOR THE WAVELETS IF YOURE USING THE OLD MOTIONMAPPER
parameters.minF = minF;
parameters.maxF = maxF;
numPerDataSet = 500; % HOW MANY SAMPLES TO GRAB FROM EACH MOVIE
nn = 95;
mD = cell(nn,2); mA = cell(nn,2);
for i = 1:nn
    i
    markers1 = allJS{i}; % THIS IS JUST THE JOINTS - FIND PROJECTIONS, MAKE FEATURES, WHATEVER
    load(['EphysBEH/dist95/info_' num2str(i) '.mat']);
    p1 = bsxfun(@minus,p1Dist',mu);
    proj = p1*vecs20;
    [data,~] = findWavelets(proj,numModes,parameters);
    n = size(p1Dist,2);
    amps = sum(data,2);
    data2 = log(data);
    data2(data2<-5) = -5;
    % JUST LOOK AT THIS - DOES doing imagesc(data2) look like a good
    % dynamic range? if not, adjust - this also applies for whatever
    % features you were using 
    % HERE I DID VELOCITY OF EACH JOINT 
    jv = zeros(n,length(xIdx));
    for j = 1:length(xIdx)
        jv(:,j) = [0; medfilt1(sqrt(sum(diff(squeeze(markers1(:,:,xIdx(j)))).^2,2)),30)];
    end
    % AND CLIPPED IT, NECESSARY FOR <TELEPORTATION>
    jv(jv>=5) = 5;
    % HERES MY FINAL TIME SERIES OF FEATURES FOR THIS MOVIE
    nnData = [data2 .1*p1z' jv];
    fprintf(1,'Running tSNE \n');
    tic
    % RUN TSNE ON IT
    yData = tsne(nnData(1:20:end,:));
    toc
    % FIND GOOD TEMPLATES AND SAVE THEM
    [signalData,signalAmps] = findTemplatesFromData(...
        nnData(1:20:end,:),yData,amps(1:20:end,:),numPerDataSet,parameters);
    mD{i} = signalData; mA{i} = signalAmps;
end

% DO BIG TSNE ON THE TEMPLATES (IF THIS IS HUGE, SUBSAMPLE DOWN TO ~40k
cD = combineCells(mD(:),1);
cA = combineCells(mA(:),1);
YD = cell(5,1);
for i = 1:5
    tic
    YD{i} = tsne(cD);
    toc
end

for i = 1:5
    ydata = YD{i};
    subplot(1,5,i);
    scatter(ydata(:,1),ydata(:,2),[],cA,'.');
end

% FIGURE OUT THE FINAL TSNE MAP - SAVE THE SAMPLE DATA AND SAMPLE EMBEDDING
ydata = YD{5};
save('EphysBEH/trainingData20220125.mat','cD','cA','ydata');


fpath = '/home/ugne/Dropbox/EphysBEH/RE_20220125/sample_';
fend = '_data_RE.mat';

fs = cell(81,1);
for i = 1:81
    fs{i} = [fpath num2str(i) fend];
end

% COMMENTED OUT HERE IS THE REEMBEDDING I PORTED TO THE CLUSTER - JUST LOAD
% THE FOLDER AND THE EMBEDDING INFO, EMBED HOWEVER IS EASIEST (I like
% Gordon's code but other distance metrics and methods might be much faster
% and easier to explain)
for i = 1:95
    i
    ma1 = allJS{i};
    save(['EphysBEH/samples95/info_' num2str(i) '.mat'],'ma1');
    
%     load(['EphysBEH/dist95/info_' num2str(i) '.mat']);
%     %p1Dist = mdist{i}; p1z = mz{i};
%     p1 = bsxfun(@minus,p1Dist',mu);
%     proj = p1*vecs20;
%     [data,~] = findWavelets(proj,numModes,parameters);
%     n = size(p1Dist,2);
%     amps = sum(data,2);
%     data2 = log(data);
%     data2(data2<-5) = -5;
%     jv = zeros(n,length(xIdx));
%     for j = 1:length(xIdx)
%         jv(:,j) = [0; medfilt1(sqrt(sum(diff(squeeze(markers1(:,:,xIdx(j)))).^2,2)),30)];
%     end
%     jv(jv>=5) = 5;
%     nnData = [data2 .1*p1z' jv];
%     %
%     save(['dannceData/asdRatData241/sample_' num2str(i) '_data.mat'],'nnData');
    % fprintf(1,'Finding Embeddings\n');
    %[zValues,zCosts,zGuesses,inConvHull,meanMax,exitFlags] = ...
    %    findTDistributedProjections_fmin(nnData,cD,...
    %    ydata,[],parameters);

%save(['quickE_asdRats/sample_' num2str(i) '_RE.mat'],'zValues','zCosts','zGuesses','inConvHull');
end

        save('dannceData/embeddingData241.mat','cD','ydata');

    save('ugneDANNCE2/trainingData20220107.mat','cD','ydata');



%% 

% FETCH THE EMBEDDINGS - COMBINE AND LOOK AT THE MAP - DO WATERSHED, how
% many regions do you have? 100 might be manageable, play with the sigma
% value in findPointDensity() 

fpath = '/home/ugne/Dropbox/EphysBEH/RE_20220125/info_';
fend = '_RE.mat';

fs = cell(95,1);
for i = 1:95
    fs{i} = [fpath num2str(i) fend];
end
%
EV = cell(95,1);
for i = 1:95
    i
    try
        % THIS ASSUMES THIS CONVEX HULL THING, YOU MIGHT NOT NEED TO DO
        % THIS
        load(fs{i});
        ev = zGuesses; ev(~inConvHull,:) = zGuesses(~inConvHull,:);
        EV{i} = ev;
    catch
    end
end

fpath = '/Users/ugne/Dropbox/dannceData/RE_20220112_2/sample_';
fend = '_RE.mat';

evall = combineCells(EV);
[xx, d] = findPointDensity(evall,1,501,[-50 50]);

 % Watershed
LL = watershed(-d,8);
LL2 = LL; LL2(d < 1e-6) = -1;
LL3 = zeros(size(LL));
for i = 1:501
    for j = 1:501
        if LL2(i,j)==0
            LL3(i,j)=1;
        end
    end
end
vSmooth = .5;
medianLength = 5;
pThreshold = [];
minRest = [];
obj = [];
fitOnly = true;
numGMM = 2;

[wr,segments,v,obj,pRest,vals,vx,vy] = ...
    findWatershedRegions_v2(evall,xx,LL,vSmooth,medianLength,pThreshold,minRest,obj,fitOnly,numGMM);

for i = 1:95
    try
[watershedRegions{i},segments,v,obj,pRest,vals,vx,vy] = ...
    findWatershedRegions_v2(EV{i},xx,LL,vSmooth,medianLength,pThreshold,minRest,obj,false,numGMM);
    catch
    end
end

[groupfs,~,~] = makeGroupsAndSegments(watershedRegions,max(max(LL)),ones(1,95),15);
% NOW YOU HAVE TIMEPOINTS WHEN EACH BEHAVIOR HAPPENED


wrh = zeros(241,125);
for i = 1:241
    wri = watershedRegions{i}; 
    wr2 = wri;
    wr2(wr2==0) = NaN;
    wr2 = fillmissing(wr2,'nearest');
    hh = hist(wr2,1:125); 
    wrh(i,:) = hh./sum(hh);
end


tt = tsne(wrh);
scatter(tt(:,1),tt(:,2),[],sampleID,'filled')





% MY CUSTOM PLOTTING CODE - LOOK AT DIEGO'S ANIMATOR REPO FOR MORE INFO ON
% HOW TO USE THAT, I FOUND IT PRETTY USEFUL - I BUILD THOSE MULTI-PANELED
% MOVIES USING THAT
offx = 250;
offy = 0;
offz = 250;
coffx = zeros(5,5); coffz = zeros(5,5);
for i = 1:5
    for j = 1:5
        coffx(i,j) = (i-1)*offx;
        coffz(i,j) = (j-1)*offz;
    end
end    
chead = [1 .6 .2]; % orange
% cspine = [198 252 3]./256; % yellow
% cspine = [252 215 3]./256; % yellow
cspine = [.2 .635 .172]; % green
cLF = [0 0 1]; % blue
cRF = [1 0 0]; % red
cLH = [0 1 1]; % cyan
cRH = [1 0 1]; % magenta
sc = [chead; chead; chead; cspine; cspine; cspine; cspine; cLF; cLF; ...
    cLF; cLF; cRF; cRF; cRF; cRF;...
    cLH; cLH; cLH; cLH; cRH; cRH; cRH; cRH];
sc1 = sc;
cgroup = [26 26 255;
    102 205 255;
    %225 179 217; %204 204 255;
    172 0 230; %152 78 163;
    102 255 153;
    %0 204 0;
    255 204 0;
    228 26 28]./256;
cgroup = cgroup([2 4 6],:);
for tm = 1:max(max(LL))
    try
        G = groups{tm};
        lgs = size(G,1);
        if lgs>20
            newG = G(randperm(lgs,20),:);
        else
            newG = G;
        end
        lenG = 60;
        gMarkers = cell(1,size(newG,1));
        for i = 1:size(newG,1)
            repG = allJJ{newG(i,1)}(newG(i,2):newG(i,3),:,:);
            if size(repG,1) > lenG
                gMarkers{i} = repG(1:lenG,:,:);
            else
                tgm = repmat(repG,[ceil(lenG/size(repG,1)) 1 1]);
                gMarkers{i} = tgm(1:lenG,:,:);
            end
        end
        allM = []; allJ = []; allC = []; allmc = [];
        for i = 1:size(newG,1)
            cx = coffx(i); cz = coffz(i);
            NM = gMarkers{i};
            NM(:,1,:) = NM(:,1,:)+cx;
            NM(:,3,:) = NM(:,3,:)+cz;
            allM = cat(3,allM,NM);
            newJ = skeleton.joints_idx+(i-1)*23;
            allJ = cat(1,allJ,newJ);
            allC = cat(1,allC,zeros(23,3));
            sccurr = repmat(cgroup(ratid(newG(i,1)),:),[23,1]);
            allmc = cat(1,allmc,sccurr);
        end
        sk2.color = allC; sk2.joints_idx = allJ; sk2.mcolor = allmc;
        
        close all;
        findic = figure('Name','Rat Test');
        h = cell(1,1);
        h{1} = Keypoint3DAnimator(allM,sk2,'Position',[0 0 1 1],'lineWidth',1);
        Animator.linkAll(h);
        set(gcf,'Units','Normalized','OuterPosition',[0.1135 0.3015 0.2289 0.5299]);
        view(h{1}.getAxes,20,30); axis equal
        axis([-200 1200 -50 100 -200 1000]);
        set(gcf,'Color','white');
        
        savePath = ['/home/ugne/Dropbox/EphysBEH/bs3/' num2str(tm) '_s.avi'];
        frames = 1:lenG;
        
        % Uncomment to write the Animation to video.
        h{1}.writeVideo(frames, savePath, 'FPS', 50, 'Quality', 70);
        
        pause(2)
    catch
    end
end

%%
for tm = 1:max(max(LL))
    try
        G = groups{tm};
        lgs = size(G,1);
        if lgs>20
            newG = G(randperm(lgs,20),:);
        else
            newG = G;
        end
        lenG = 60;
        gMarkers = cell(1,size(newG,1));
        for i = 1:size(newG,1)
            repG = allJZ{newG(i,1)}(newG(i,2):newG(i,3),:,:);
            if size(repG,1) > lenG
                gMarkers{i} = repG(1:lenG,:,:);
            else
                tgm = repmat(repG,[ceil(lenG/size(repG,1)) 1 1]);
                gMarkers{i} = tgm(1:lenG,:,:);
            end
        end
        allM = []; allJ = []; allC = []; allmc = [];
        for i = 1:size(newG,1)
            cx = coffx(i); cz = coffz(i);
            NM = gMarkers{i};
            NM(:,1,:) = NM(:,1,:);
            NM(:,3,:) = NM(:,3,:);
            allM = cat(3,allM,NM);
            newJ = skeleton.joints_idx+(i-1)*23;
            allJ = cat(1,allJ,newJ);
            allC = cat(1,allC,zeros(23,3));
            sccurr = repmat(cgroup(ratid(newG(i,1)),:),[23,1]);
            allmc = cat(1,allmc,sccurr);
        end
        sk2.color = allC; sk2.joints_idx = allJ; sk2.mcolor = allmc;
        
        close all;
        findic = figure('Name','Rat Test');
        h = cell(1,1);
        h{1} = Keypoint3DAnimator(allM,sk2,'Position',[0 0 1 1],'lineWidth',1);
        Animator.linkAll(h);
        set(gcf,'Units','Normalized','OuterPosition',[0.1135 0.3015 0.2289 0.5299]);
        view(h{1}.getAxes,20,30); axis equal
        axis([-300 500 -300 500 -100 400]);
        set(gcf,'Color','white');
        
        savePath = ['/home/ugne/Dropbox/EphysBEH/bs1/' num2str(tm) '_s.avi'];
        frames = 1:lenG;
        
        % Uncomment to write the Animation to video.
        h{1}.writeVideo(frames, savePath, 'FPS', 50, 'Quality', 70);
        
        pause(2)
    catch
    end
end




%% sort by com/ joints 5+6?
V = cell(241,1);
for i = 1:241
    temp = allFS{i}(:,:,5);
    td = [0 0 0; diff(temp)];
    tmov = sqrt(sum(td.^2,2));
    tmd = smooth(medfilt1(tmov,10),50);
    V{i} = tmd;
end

allV = combineCells(V);
allWR = combineCells(watershedRegions);
for b = 1:125
    fx = find(allWR==b);
    bv = allV(fx);
    medV(b) = median(bv);
    med25V(b) = prctile(bv,25);
    med75V(b) = prctile(bv,75);
end

[~,sid] = sort(medV);
plot(medV(sid))
hold on; plot(med25V(sid));
hold on; plot(med75V(sid));


%% listing
% language for postures 



l100{1} = 'reared - up';
l100{2} = 'reared - up';
l100{3} = 'reared - up';
l100{4} = 'rearing - twist, some error';
l100{5} = 'reared';
l100{6} = 'reared';
l100{7} = '';
l100{8} = '';
l100{9} = 'crouched';
l100{10} = 'crouched';

l100{11} = 'idle';
l100{12} = '';
l100{13} = 'rouched';
l100{14} = 'locomotion - head down';
l100{15} = 'idle';
l100{16} = '';
l100{17} = '';
l100{18} = 'error - maybe COM?';
l100{19} = 'step - head up';
l100{20} = '';

l100{21} = '';
l100{22} = 'step crouch up';
l100{23} = 'nose tracking error, mostly idle';
l100{24} = 'full body mov';
l100{25} = '';
l100{26} = 'turn crouched';
l100{27} = 'body mov - extension';
l100{28} = 'idle';
l100{29} = 'idle';
l100{30} = 'idle';

l100{31} = 'body extend small';
l100{32} = 'anterior some groom';
l100{33} = 'idle';
l100{34} = 'idle';
l100{35} = 'idle - head up';
l100{36} = 'idle - sleeping pose';
l100{37} = 'idle - standing pose';
l100{38} = 'body mov - small extension';
l100{39} = 'error - tracking';
l100{40} = 'head swing small';

l100{41} = 'idle - scrunched';
l100{42} = 'idle - flat';
l100{43} = 'idle';
l100{44} = 'groom - face and hands small';
l100{45} = 'idle';
l100{46} = 'idle - elongated';
l100{47} = 'head sweep, some error';
l100{48} = 'head up - mixed pose';
l100{49} = 'idle';
l100{50} = 'idle or very small';

l100{51} = 'idle';
l100{52} = 'small, some error';
l100{53} = 'head sweep - up';
l100{54} = 'idle - scrunched';
l100{55} = 'idle or very small';
l100{56} = 'idle - tracking blip error';
l100{57} = 'idle';
l100{58} = 'anterior small';
l100{59} = 'head up';
l100{60} = 'idle';

l100{61} = 'error tracking';
l100{62} = 'idle error tracking';
l100{63} = 'idle some error';
l100{64} = 'idle some error tracking';
l100{65} = 'idle';
l100{66} = 'idle';
l100{67} = 'small anterior';
l100{68} = 'error tracking';
l100{69} = '';
l100{70} = 'small scrunched';

l100{71} = 'error tracking';
l100{72} = 'idle or small';
l100{73} = 'idle - flat posture';
l100{74} = 'idle - upright';
l100{75} = 'idle ';
l100{76} = 'idle ';
l100{77} = 'idle some tracking error';
l100{78} = 'small anterior';
l100{79} = 'small anterior';
l100{80} = 'idle';

l100{81} = 'idle';
l100{82} = 'idle or very small';
l100{83} = 'idle tracking error';
l100{84} = 'idle';
l100{85} = 'idle';
l100{86} = 'anterior small';
l100{87} = 'idle';
l100{88} = 'idle';
l100{89} = 'idle';
l100{90} = 'idle - laying down';

l100{91} = 'small head sweep';
l100{92} = 'idle - laying down';
l100{93} = 'idle';
l100{94} = 'head sweep/ rotate';
l100{95} = 'small - scrunched';
l100{96} = 'idle - flat';
l100{97} = 'idle';
l100{98} = 'small head move - up';
l100{99} = 'error';
l100{100} = '';




% some metrics for each group - 
% word description + purity measure by:

% stereotypy, duration, 
% some mixing comes from joint errors
% tracking capacity - better proofreading and interpolation




%%

% clear


load('foldernames.mat')
ff = allNames(1:95);

jAll = cell(95,1);
for i = 1:95
    i
    clear ma1
    load(['samples95/info_' num2str(i) '.mat']);
    jAll{i} = ma1;
end


EV = cell(95,1);
for i = 1:95
    i
    try
        load(['RE_20220201/info_' num2str(i) '_RE.mat']);
        ev = zGuesses; ev(~inConvHull,:) = zGuesses(~inConvHull,:);
        EV{i} = ev;
    catch
    end
end



evall = combineCells(EV);
[xx, d] = findPointDensity(evall,1.2,501,[-68 68]);

 % Watershed
LL = watershed(-d,8);
LL2 = LL; LL2(d < 1e-6) = -1;
LL3 = zeros(size(LL));
for i = 1:501
    for j = 1:501
        if LL2(i,j)==0
            LL3(i,j)=1;
        end
    end
end
vSmooth = .5;
medianLength = 5;
pThreshold = [];
minRest = [];
obj = [];
fitOnly = true;
numGMM = 2;

[wr,segments,v,obj,pRest,vals,vx,vy] = ...
    findWatershedRegions_v2(evall,xx,LL,vSmooth,medianLength,pThreshold,minRest,obj,fitOnly,numGMM);

for i = 1:95
    try
[watershedRegions{i},segments,v,obj,pRest,vals,vx,vy] = ...
    findWatershedRegions_v2(EV{i},xx,LL,vSmooth,medianLength,pThreshold,minRest,obj,false,numGMM);
    catch
    end
end

[groups,~,~] = makeGroupsAndSegments(watershedRegions,max(max(LL)),ones(1,95),15);

jt = jAll{2};
evt = EV{2};
[~,d2] = findPointDensity(evt,1.2,501,[-68 68]);
scatter(evt(:,1),evt(:,2),[],1:length(evt),'.')

idx = 1:50000;
scatter(evt(idx,1),evt(idx,2),[],1:length(evt(idx,:)),'.')
hold on; 
plot(evt(idx,1),evt(idx,2))
