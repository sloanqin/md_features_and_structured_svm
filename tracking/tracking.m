%% DEMO_TRACKING
%
% Running the MDNet tracker on a given sequence.
%
% Hyeonseob Nam, 2015
%

clear;
otb50 = importdata('./dataset/OTB50.txt');
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
    result = mdstruck_run(conf.imgList, conf.gt(1,:), net);
    %result = mdnet_run(conf.imgList, conf.gt(1,:), net);
    
    save(['./dataset/result/',seqname,'_result.mat'],'result');
end
