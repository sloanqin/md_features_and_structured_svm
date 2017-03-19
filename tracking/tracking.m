%% DEMO_TRACKING
%
% Running the MDNet tracker on a given sequence.
%
% Hyeonseob Nam, 2015
%

clear;
otb50 = importdata('./dataset/OTB50_4.txt');
algorithmName = 'MDStruck';
for i=1:size(otb50,1)
    seqname = otb50{i,1};
    conf = genConfig('otb',seqname);
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
end
