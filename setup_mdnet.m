%% SETUP_MDNET
%
% Setup the environment for running MDNet.
%
% Hyeonseob Nam, 2015 
%

if(isempty(gcp('nocreate')))
    parpool;
end

run matconvnet/matlab/vl_setupnn ;

%qyy,add path
addpath('pretraining');
addpath('tracking');
addpath('utils');
addpath('tracking');
