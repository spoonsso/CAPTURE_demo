%% make a multicolor scatter

%animalID = cat(1,ones(1296000/4,1),2*ones(1296000/4,1),3*ones(1296000/4,1),4*ones(1296000/4,1));
animalID = cat(1,ones(2268000/7,1),2*ones(2268000/7,1),3*ones(2268000/7,1),4*ones(2268000/7,1),2*ones(2268000/7,1),3*ones(2268000/7,1),4*ones(2268000/7,1));

aID = repelem(animalID,3,1);
aIDinds = aID(analysisstruct.frames_with_good_tracking{1});

colors = lines(4);

names = {'dopa','healthy','dart','lesion'};

figure('Position',[400,400,1000,1000]); hold on;
for i = [2,4,1,3]
    scatter(analysisstruct.zValues(aIDinds==i,1),analysisstruct.zValues(aIDinds==i,2),5,colors(i,:),'filled','markerfacealpha',0.5,'DisplayName',names{i});
    %plot1.Color(4) = 0.2;
end
xlabel('tSNE1');
ylabel('tSNE2');
axis('square')
title('PDb 1st + 2nd hour co-embedded [no dopa 2nd hour]');
legend('AutoUpdate','off')

nnn = analysisstruct.sorted_watershed;
nnn(nnn>0) = 1;
B = bwboundaries(((nnn)));
%figure(333)
hold on
for kk = 1:numel(B)
    if numel(find(ismember(analysisstruct.sorted_clust_ind,kk)))
        if numel(find(analysisstruct.annot_reordered{end,end}==find(analysisstruct.sorted_clust_ind==kk)))>1
            plot(analysisstruct.xx(B{kk}(:,2)),analysisstruct.yy(B{kk}(:,1)),'k')
        end
    end
end
%% show significant watersheds per condition relative to baseline (healthy).

% do for baseline
nclust = numel(unique(analysisstruct.annot_reordered{1}));

clustrx = zeros(4,nclust);

% count the number of behaviors in each cluster for each condition
for i =1:nclust
    for j = 1:4
        clustrx(j,i) = numel(find((analysisstruct.annot_reordered{1}==i) & (aIDinds==j)'));
    end
end
%% Get poisson tail probabilities using poisscdf
% need to decide what tail to take based on whether it is a postive or negative fold change

wt = 2;
% do dopa v healthy
rx = 3;
dopa_p = [];
shading = [];
for i=1:nclust
    if clustrx(rx,i) <= clustrx(wt,i)
        tp = poisscdf(clustrx(rx,i),clustrx(wt,i));
    else
        tp = 1-poisscdf(clustrx(rx,i),clustrx(wt,i));
    end
    
    %     if clustrx(rx,i) == 0 && clustrx(wt,i) == 0
    %         tp =1
    %     end
    %     if tp == 0
    %         keyboard;
    %     end
    dopa_p = [dopa_p tp];
    shading = [shading clustrx(rx,i)/clustrx(wt,i)];
end

shading(shading>=64) = 64;
shading(shading<=(1/64)) = 1/64;
shading(dopa_p>=1e-6/numel(dopa_p)) = NaN;
% now drawt he watersheds and color them according to fodl change, but only
% coloring ones with sig. p-values

dens = sum(clustrx,1)/sum(clustrx(:));
dens_abs = dens*sum(clustrx(:));

shading(dens_abs<120) =NaN;
% dens = (dens-min(dens))/(max(dens)-min(dens));
%
% dens(dens>=0.7) = 0.7;
% dens = (dens-min(dens))/(max(dens)-min(dens));

colors = cat(1,0.95*ones(3,3),othercolor('PuRd9',101));

% shading_values = 100.*(shading-[min(shading)+0.001])./(max(shading)-[min(shading)+0.001]);
% shading_values(shading_values<0) = 0;

shading_values = shading;

figure;
nnn = analysisstruct.sorted_watershed;%analysisstruct.unsorted_watershed;
nnn(nnn>0) = 1;
B = bwboundaries((flipud(nnn)));
hold on
for kk = 1:numel(B)
    if kk<=numel(shading_values)
        kkhere = (analysisstruct.sorted_clust_ind(kk));
        %         fill(analysisstruct.xx(B{kkhere}(:,2)),analysisstruct.yy(numel(analysisstruct.yy)-B{kkhere}(:,1)),colors(1+floor(shading_values(kk)),:),...
        %             'EdgeColor',colors(1+floor(shading_values(kk)),:),'Linewidth',2); %,'none'
        h= fill(analysisstruct.xx(B{kkhere}(:,2)),analysisstruct.yy(numel(analysisstruct.yy)-B{kkhere}(:,1)),log(shading_values(kk)));
        colormap(usa_divergent);
        %set(h,'facealpha',dens(kk))%,'none'
    end
end
%% just do separate density maps
analysisstruct.condition_inds = aIDinds;
%   zvals_cell_array = cell(1,max(unique(analysisstruct.condition_inds)));
lup_conds = {[2,4],[2,1],[2,3]};
%clrs = {'Blues9','Purples9','Greens9','Reds9'};
clrs = {white_cyan,white_gray,white_red2,'Reds9'};
xclrnames = {'gray','red'};
for cc =1%:numel(lup_conds)
    zvals_cell_array = {};
    lup = lup_conds{cc};
    for ll = [1,2]%unique(analysisstruct.condition_inds)'
        zvals_cell_array{ll} = analysisstruct.zValues(find(analysisstruct.condition_inds==lup(ll)),:);
    end
    
    
    %     badclust = find(cellfun(@numel,strfind(analysisstruct.clusternames,'BadTracking')));
    %   badframes = find(ismember(analysisstruct.annot_reordered{end,end},badclust));
    %   goodframes = setxor(1:numel(analysisstruct.annot_reordered{end,end}),badframes);
    %   if numel(zvals_cell_array) == 1 && numel(badclust)>1
    %       zvals_cell_array{1} = zvals_cell_array{1}(goodframes,:);
    %   end
    %h3 = subplot(1,1,1)
    h2 = figure(444);
    fighand_in = h2;
    set(h2,'Color','w')
    % plotdensitymaps({cat(1,zvals_cell_array{:})},1,fighand_in,analysisstruct.params.density_width,...
    %     max(analysisstruct.zValues(:))*analysisstruct.params.expansion_factor,analysisstruct.params.density_res)
    % threshcols = {};
    % threshcols{1} = purpleThresh;
    % threshcols{2} = greenThresh;
    plotdensitymaps(zvals_cell_array,1,fighand_in,analysisstruct.params.density_width,...
        max(analysisstruct.zValues(:))*analysisstruct.params.expansion_factor,analysisstruct.params.density_res,clrs(lup))%, threshcols)
    
    figure(480);
    nnn = flipud(analysisstruct.sorted_watershed);
    nnn(nnn>0) = 1;
    B = bwboundaries(((nnn)));
    %figure(333)
    hold on
    for kk = 1:numel(B)
        if numel(find(ismember(analysisstruct.sorted_clust_ind,kk)))
            if numel(find(analysisstruct.annot_reordered{end,end}==find(analysisstruct.sorted_clust_ind==kk)))>1
                plot(analysisstruct.xx(B{kk}(:,2))*6.35+500,analysisstruct.yy(B{kk}(:,1))*6.35+500,'k')
            end
        end
    end
    
    if ~ischar(clrs{lup(1)})
        clr1 = xclrnames{1};
    else
        clr1 = clrs{lup(1)};
    end
    
    if ~ischar(clrs{lup(2)})
        clr2 = xclrnames{2};
    else
        clr2 = clrs{lup(2)};
    end
    axis('square');
    title([names{lup(1)} '(' clr1 ')-' names{lup(2)} '(' clr2 ')'])
    % print('-dpng',[names{lup(1)} '_' names{lup(2)} '.png'],'-r1200')
    % print('-depsc',[names{lup(1)} '_' names{lup(2)} '.eps'],'-r1200')
    
    figure(99)
    axis('square');
    title(names{lup(1)});
    % print('-dpng',[names{lup(1)} '.png'],'-r1200')
    % print('-depsc',[names{lup(1)} '.eps'],'-r1200')
    %
    figure(100)
    axis('square');
    title(names{lup(2)});
    % print('-dpng',[names{lup(2)} '.png'],'-r1200')
    % print('-depsc',[names{lup(2)} '.eps'],'-r1200')
end
%%
% divide all counts by 2 except for dopa, because we are not using the
% second hour dopa and need rates to be over same time window
clustrx(2:end,:)=round(clustrx(2:end,:)/2);
%% Get poisson tail probabilities using poisscdf
% need to decide what tail to take based on whether it is a postive or negative fold change

wt = 2;
% get unadjusted p-values for everything compared to healthy

allp = {};
allshade = {};
allsig = {};
for rx = [1,3,4]
    dopa_p = [];
    shading = [];
    for i=1:nclust
        if clustrx(rx,i) <= clustrx(wt,i)
            tp = poisscdf(clustrx(rx,i),clustrx(wt,i));
        else
            tp = 1-poisscdf(clustrx(rx,i),clustrx(wt,i));
        end
        
        if clustrx(rx,i) == 0 && clustrx(wt,i) == 0
            tp =1;
        end
        
        dopa_p = [dopa_p tp];
        shading = [shading clustrx(rx,i)/clustrx(wt,i)];
    end
    allp{rx} = dopa_p;
    allshade{rx} = shading;
    allsig{rx} = dopa_p<=1e-6/(3*numel(dopa_p));
end

% then do sig different from lesion
wt = 4;
% get unadjusted p-values for everything compared to healthy

allp_les = {};
allshade_les = {};
allsig_les = {};
for rx = [1,3]
    dopa_p = [];
    shading = [];
    for i=1:nclust
        if clustrx(rx,i) <= clustrx(wt,i)
            tp = poisscdf(clustrx(rx,i),clustrx(wt,i));
        else
            tp = 1-poisscdf(clustrx(rx,i),clustrx(wt,i));
        end
        
        if clustrx(rx,i) == 0 && clustrx(wt,i) == 0
            tp =1;
        end
        
        dopa_p = [dopa_p tp];
        shading = [shading clustrx(rx,i)/clustrx(wt,i)];
    end
    allp_les{rx} = dopa_p;
    allshade_les{rx} = shading;
    allsig_les{rx} = dopa_p<=1e-6/(3*numel(dopa_p));
end

%% Now find all downregulated lesion that is signficant
% of those, which were differentially rescued by dopa and dart?
lost = allshade{4}<1 & allsig{4}==1;
gain_dopa = allshade{1}>1 & allsig{1}==1;
gain_dart = allshade{3}>1 & allsig{3}==1;
% gain_dopa = allsig{1}==0;
% gain_dart = allsig{3}==0;
gain_dopa = gain_dopa | allsig{1}==0;
gain_dart = gain_dart | allsig{3}==0;
% gain_dopa = allshade_les{1}>1 & allsig_les{1}==1;
% gain_dart = allshade_les{3}>1 & allsig_les{3}==1;
pop1 = lost & gain_dopa & ~gain_dart;
pop2 = lost & ~gain_dopa & gain_dart;
pop = pop1 | pop2;
%% Plot these on map
figure;
nnn = analysisstruct.sorted_watershed;%analysisstruct.unsorted_watershed;
nnn(nnn>0) = 1;
B = bwboundaries((flipud(nnn)));
hold on
for kk = 1:numel(B)
    if kk<=numel(shading)
        kkhere = (analysisstruct.sorted_clust_ind(kk));
        if kk==63%pop(kk)
            h= fill(analysisstruct.xx(B{kkhere}(:,2)),analysisstruct.yy(numel(analysisstruct.yy)-B{kkhere}(:,1)),'k');
            %%colormap(usa_divergent);
            set(h,'facealpha',0.5)%,'none'
        end
    end
end

nnn = flipud(analysisstruct.sorted_watershed);
nnn(nnn>0) = 1;
B = bwboundaries(flipud(nnn));
%figure(333)
hold on
for kk = 1:numel(B)
    if numel(find(ismember(analysisstruct.sorted_clust_ind,kk)))
        if numel(find(analysisstruct.annot_reordered{end,end}==find(analysisstruct.sorted_clust_ind==kk)))>1
            plot(analysisstruct.xx(B{kk}(:,2)),analysisstruct.yy(B{kk}(:,1)),'k')
        end
    end
end
xlim([-80,80])
ylim([-80,80])
axis('square');
%% To inspect wire frames...first plot the above watershed/
% then run this to get the indices you need for a single area
[x, y] = getline(gcf);
  IN = inpolygon(analysisstruct.zValues(:,1),analysisstruct.zValues(:,2),...
        x ,y);
%%
analysisstruct.mocapstruct_reduced_agg{1}.mocapfiletimes{1} = [];
h2 =figure(1)
buffer = [-10:10]
beh_list = examine_features(h2,analysisstruct.zValues,...
analysisstruct.subset_of_points_to_plot_tsne_capped,...
analysisstruct.condition_inds, analysisstruct.subset_of_points_to_plot_tsne_capped ,...
analysisstruct.mocapstruct_reduced_agg,1,[1],analysisstruct.conditionnames,buffer,IN);
    