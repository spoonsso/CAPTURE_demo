function [links,colors] = load_link_files(linkname)
% this file loads the links for pre-defined skeletons made using DANNCE
% Rats - the 20 marker CAPTURE dataset
% Mouse - the 14 marker low-res mouse recordings
% kylemouse - the 20 marker kyle skeleton


switch linkname
    
    case 'rats'
        
        colors = {'b','b','b',...
            'r','r','r',...
            'm','m',...
            'c','g',... %hips
            'y','y','y',... %L arm
            'w','w','w',... %R arm
            'g','c',...
            'c','g','k','k','k',};
        
        links = {[1,2],[2,3],[1,3],...
            [2,4],[1,4],[3,4],... %head to spine
            [4,5],[5,6],...
            [4,7],[7,8],[5,8],[5,7],[6,7],...
            [6,9],[6,10],...
            [11,12],[4,13],[4,14],[11,13],[12,13],...
            [14,15],[14,16],[15,16],...
            [9,18],[10,17],...%knees
            [18,19],[17,20]};
        
        colors{14} = [171 142 73]./255;
        colors{15} = [171 142 73]./255;
        colors{16} = [171 142 73]./255;
        links{20} = [];
        links{22} = [];
        
    case 'mouse'
        links =  {[1 2],[2 3],[1 3],[2 4],[4 5],[5 6],[7 8],[ 8 4],[9 10],[ 10 4],[11 12],[12 6],[13 14],[14 6]};
        
        colors  = {'b','b','b',...
            'r','r','r',...
            'y','y',... %hips
            [171 142 73]./255,[171 142 73]./255,... %L arm
            'g','g',... %R arm
            'c','c',...
            'm','m','k','k','k'};
        
    case 'pups'
        links =  {[1 2],[1 3],[3 2],[1 4],[4 2],[4 3],[4 5],[5 6],[7 6],[7 8],...
            [4 10],[10 9],[4 12],[12 11],[6 14],[6 16],[14 13],[15 16]};%
        
        colors  = {'b','b','b',...
            'r','r','r',...
            'm','m',... %tail
            [171 142 73]./255,[171 142 73]./255,... %L arm
            'g','g',... %R arm
            'c','c',... %LHL
            'y','y','k','k','k'}; %RHL
        
    case 'marmoset'
        links =  {[1 2],[2 3],[1 3],[2 4],[4 5],[5 6],[7 8],[4 8],[9 10],[4 10],[11 12],[12 6],[13 14],[14 6],[6 15],[15 16]};
        
        
        colors  = {'b','b','b',...
            'r','r','r',...
            'y','y',... %hips
            [171 142 73]./255,[171 142 73]./255,... %L arm
            'g','g',... %R arm
            'c','c',...
            'm','m','k','k','k'};
        
    case 'kyle_mouse'
        links =  {[1 2],[2 3],[1 3],[2 4],[4 5],[5 6],[7 8],[4 8],[9 10],[4 10],[12 6],...
            [14 6],[7,15],[4,16],[16,8],[4,18],[9 17],[16 10],[21 6],[19 12]   ,[13 20],[20 14],[11 19]};
        %[11 12], [13 14],
        
        colors  = {'b','b','b',...
            'r','r','r',...
            'y','y',... %Larm
            [171 142 73]./255,[171 142 73]./255,... %R arm
            'g','g',... %L leg
            'c','c',...%R leg
            'y','y',... %Larm
            [171 142 73]./255,[171 142 73]./255,... % R arm
            'g','c',...
            'm','m','k'};
    case 'bird'
        birdlinks = load('X:\Jesse\MotionAnalysisCaptures\DANNCE_animals\manuscript_formattedData\bird\bird18.mat');
         links =birdlinks.joints_idx;
            colors =birdlinks.color;
      colors = num2cell(colors,2);
links = num2cell(links,2);
    case 'taddy_mouse'
          taddylinks = load('/hpc/group/tdunn/joshwu/CAPTURE_demo/Species_specific_files/left_or_right_colormap.mat');
         links =taddylinks.joints_idx;
            colors =taddylinks.color;
      colors = num2cell(colors,2);
links = num2cell(links,2);
        
end

