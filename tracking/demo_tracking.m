%% DEMO_TRACKING
%
% Running the MDNet tracker on a given sequence.
%
% Hyeonseob Nam, 2015
%

clear;

algorithmName = 'MDStruck';
%algorithmName = 'MDNet';
seqname = 'singer2';
conf = genConfig('otb',seqname);
% conf = genConfig('vot2015','ball1');

switch(conf.dataset)
    case 'otb'
        net = fullfile('models','mdnet_vot-otb.mat');
    case 'vot2014'
        net = fullfile('models','mdnet_otb-vot14.mat');
    case 'vot2015'
        net = fullfile('models','mdnet_otb-vot15.mat');
end

res = mdstruck_run(conf.imgList, conf.gt(1,:), net);
%res = mdnet_run(conf.imgList, conf.gt(1,:), net);

%% save result to .mat file
result.res = res;
result.fps = -1;
result.len = length(conf.imgList);
result.annoBegin = 1;
result.startFrame = 1;
result.anno = conf.gt;
result.type = 'rect';
	
if(strcmp(seqname,'david'))
	result.annoBegin = 300;
	result.startFrame = 300;
end
    
results{1,1} = result;
save(['./dataset/result/' seqname '_' algorithmName '.mat'],'results');
