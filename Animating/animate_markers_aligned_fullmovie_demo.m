function M = animate_markers_aligned_fullmovie_demo(mocapstruct,frame_inds,fighand)
%matlab_fr = 10;
if nargin<3
    h=figure;%(370)
else
    h=fighand;
end

frame_last = 0;

marker_plot = ones(1,numel(mocapstruct.markernames));


%% initialize the figure
% set(h,'Color','k')

xx = squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{1})(1,1));
yy = squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{1})(1,2));
zz = squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{1})(1,3));
handle_base = line(xx,yy,zz,'Marker','o','Color',mocapstruct.markercolor{1},'MarkerFaceColor',mocapstruct.markercolor{1},'MarkerSize',6);


ax = gca;
axis(ax,'manual')
%set(gca,'Color','k')
grid off;
% set(gca,'Xcolor',[1 1 1 ]);
% set(gca,'Ycolor',[1 1 1]);
% set(gca,'Zcolor',[1 1 1]);

zlim([-110 170])
xlim([-140 140])
ylim([-140 140])

%     zlim([-210 270])
%     xlim([-240 240])
%     ylim([-240 240])
%

set(gca,'XTickLabels',[],'YTickLabels',[],'ZTickLabels',[])
view([-22, 12]);
fn = fieldnames(mocapstruct.markers_aligned_preproc);
d3da = zeros(size(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{1}),1),3,numel(mocapstruct.markernames));
for i = 1:numel(fn)
    d3da(:,:,i) = mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{i});
end

%     mcolor_ = load('/media/twd/dannce-pd/PDBehavior_dannce_v2/left_or_right_colormap.mat');
%     mcolor = zeros(numel(mcolor_.joint_names),3);
%     for i = 1:size(mcolor_.joints_idx)
%         mcolor(i,:) = mcolor_.color(mcolor_.joints_idx(i,2),:);
%     end
%     

ln = lines(6);
    mcolor = zeros(numel(fn),3);
%     mcolor(1,:) = [0 0 1];
%     mcolor(2,:) = [0 0 1];
%     mcolor(3,:) = [0 0 1];
%     mcolor(4,:) = [1 0 0];
%     mcolor(5,:) = [1 0 0];
%     mcolor(6,:) = [1 0 0];
%     mcolor(7,:) = [1 0 0];
%     mcolor(8,:) = [1 0 0];
%     mcolor(9,:) = [1 0 1];
%     mcolor(10,:) = [1 0 1];
%     mcolor(11,:) = [1 0 1];
%     mcolor(12,:) = [1 0 1];
%     mcolor(13,:) = [1 0 1];
%     mcolor(14,:) = [1 0 1];
%     mcolor(15,:) = [0 1 0];
%     mcolor(16,:) = [0 1 0];
%     mcolor(17,:) = [0 1 0];
%     mcolor(18,:) = [0 1 0];
%     mcolor(19,:) = [0 1 0];
%     mcolor(20,:) = [0 1 0];
    mcolor(1,:) = ln(1,:);
    mcolor(2,:) = ln(1,:);
    mcolor(3,:) = ln(1,:);
    mcolor(4,:) = ln(2,:);
    mcolor(5,:) = ln(2,:);
    mcolor(6,:) = ln(2,:);
    mcolor(7,:) = ln(2,:);
    mcolor(8,:) = ln(2,:);
    mcolor(9,:) = ln(3,:);
    mcolor(10,:) = ln(3,:);
    mcolor(11,:) = ln(3,:);
    mcolor(12,:) = ln(3,:);
    mcolor(13,:) = ln(3,:);
    mcolor(14,:) = ln(3,:);
    mcolor(15,:) = ln(5,:);
    mcolor(16,:) = ln(5,:);
    mcolor(17,:) = ln(5,:);
    mcolor(18,:) = ln(5,:);
    mcolor(19,:) = ln(5,:);
    mcolor(20,:) = ln(5,:);
    
% convert
for lk = reshape(frame_inds,1,[])%1:10:10000
    fprintf('frame %f \n',lk);
    cla;
    
    
    %mocapstruct.links{20} = [];
    %       mocapstruct.links{22} = [];
    
    ind_to_plot = lk;
    
    d3d = squeeze(d3da(ind_to_plot,:,:));
    [az,el,r] = cart2sph(d3d(1,4),d3d(2,4),d3d(3,4));
    
    % for rearing frames?
%     az = 50*pi/180;
%     el = 65*pi/180;

%     az = -50*pi/180;
%     el = 65*pi/180;
    
    
    % az rotation
    azr = rotz(-az*180/pi);
    elr = roty(-el*180/pi);
    
    rotd3d = azr*d3d;
    rotd3d = elr*rotd3d;
   rotd3d = rotx(-45)*rotd3d;
    %rotd3d = rotx(0)*rotd3d;
    

    
    %% Plot markers that are tracked in the frame
    set(gca,'Nextplot','ReplaceChildren');
    handles_here = cell(1,numel(mocapstruct.markernames));
    for jj = 1:numel(mocapstruct.markernames)
        % don't plot markers that drop out
        if ~isnan(sum(mocapstruct.markers_preproc.(mocapstruct.markernames{jj})(ind_to_plot,:),2))
            if (~sum(mocapstruct.markers_preproc.(mocapstruct.markernames{jj})(ind_to_plot,:),2) == 0) && jj ~= 8 && jj ~=8
                %             xx = squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{jj})(ind_to_plot,1));
                %                 yy = squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{jj})(ind_to_plot,2));
                %                 zz = squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{jj})(ind_to_plot,3));
                
                %figure(3);hold off;
                %plot3(rotd3d(1,:), rotd3d(2,:), rotd3d(3,:),'.r')
                xx = squeeze(rotd3d(1,jj));
                yy = squeeze(rotd3d(2,jj));
                zz = squeeze(rotd3d(3,jj));
                handles_here{jj} = line(xx,yy,zz,'Marker','o','Color',mcolor(jj,:),'MarkerFaceColor',mcolor(jj,:),'MarkerSize',4);
%                 handles_here{jj} = line(xx,yy,zz,'Marker','o','Color',mocapstruct.markercolor{jj},'MarkerFaceColor',mocapstruct.markercolor{jj},'MarkerSize',3);
                
                
                
                hold on
                marker_plot(jj) = 1;
            else
                marker_plot(jj) = 0;
            end
            
        end
    end
    
    %% plot the links between markers
    for mm = 1:numel(mocapstruct.links)
        if numel(mocapstruct.links{mm})
            if (ismember(mocapstruct.links{mm}(1),1:numel(mocapstruct.markernames)) && ismember(mocapstruct.links{mm}(2),1:numel(mocapstruct.markernames))) && mocapstruct.links{mm}(1) ~= 8 mocapstruct.links{mm}(2) ~= 8
                if (marker_plot(mocapstruct.links{mm}(1)) == 1 && marker_plot(mocapstruct.links{mm}(2)) == 1)
                    
                    %                     xx = [squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{mocapstruct.links{mm}(1)})(ind_to_plot,1)) ...
                    %                         squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{mocapstruct.links{mm}(2)})(ind_to_plot,1)) ];
                    %                     yy = [squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{mocapstruct.links{mm}(1)})(ind_to_plot,2)) ...
                    %                         squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{mocapstruct.links{mm}(2)})(ind_to_plot,2))];
                    %                     zz = [squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{mocapstruct.links{mm}(1)})(ind_to_plot,3)) ...
                    %                         squeeze(mocapstruct.markers_aligned_preproc.(mocapstruct.markernames{mocapstruct.links{mm}(2)})(ind_to_plot,3))];
                    
                    xx = [squeeze(rotd3d(1,mocapstruct.links{mm}(1))),squeeze(rotd3d(1,mocapstruct.links{mm}(2)))];
                    yy = [squeeze(rotd3d(2,mocapstruct.links{mm}(1))),squeeze(rotd3d(2,mocapstruct.links{mm}(2)))];
                    zz = [squeeze(rotd3d(3,mocapstruct.links{mm}(1))),squeeze(rotd3d(3,mocapstruct.links{mm}(2)))];
                    line(xx,yy,zz,'Color','k','LineWidth',0.8)%mocapstruct.markercolor{mocapstruct.links{mm}(1)},'LineWidth',3);
                end
                
            end
        end
    end
    
    view([5.1,12]);
    title(num2str(lk));
    drawnow
    
    hold off
    %keyboard;
    
    frame_last = lk;
    
    M(find(frame_inds == lk)) =  getframe(gcf);
    %waitforbuttonpress;
    
    %clf
end
%set(gca,'Nextplot','add');

end