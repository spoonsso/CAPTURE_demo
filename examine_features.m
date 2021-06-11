function [beh_list,behlist_inst,zvaltot] = examine_features(fighandle,zValues,subset_of_points_to_plot,cond_inds,good_tracks,mocapstructs,videoflag,condsuse,conditionname,buffer,IN)
 
if nargin < 11
[x, y] = getline(fighandle);
  IN = inpolygon(zValues(:,1),zValues(:,2),...
        x ,y);
end
    zvaltot = IN;
    beh_list = cell(1,numel(mocapstructs));
        behlist_inst = cell(1,numel(mocapstructs));

    maxinds = find(cellfun(@numel,good_tracks));
    for kk = condsuse
    [~,ia] = intersect(find(cond_inds==kk),find(IN));
    ia = subset_of_points_to_plot{kk}(ia);
    behlist_inst{kk} = unique(rectify_inds(bsxfun(@plus,reshape(good_tracks{kk}(ia),1,[])',buffer),max(good_tracks{kk})));
     beh_list{kk} = sort(reshape(unique(rectify_inds(bsxfun(@plus,reshape(good_tracks{kk}(ia),1,[])',buffer),max(good_tracks{kk}))),1,[]),'ascend');    
    end
    
    maxval = 1000;
    
    h3=figure(390);
    close 390
    
    
% h3=figure(390);
%     set(h3,'Color','k')
% 
%     for kk = 1:numel(condsuse)%maxinds
%         h{kk}=figure(390);
% 
% %h{kk} = subplot(1,numel(condsuse),kk);
%                     animate_markers_timelapse(mocapstructs{condsuse(kk)},beh_list{condsuse(kk)}(1:end),h{kk});
%                     ntitle(conditionname{condsuse(kk)},'color','w')
%     end
    
    if (videoflag)
               h3=figure(370);
         for kk = condsuse
          animate_markers_aligned_fullmovie_demo(mocapstructs{kk},beh_list{kk}(1:min(maxval,numel(beh_list{kk}))));
         end
    end
end