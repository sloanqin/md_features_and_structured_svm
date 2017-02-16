function [ feat ] = mdnet_features_fc4(net, ims, opts)
% MDNET_FEATURES_FCX
% Compute CNN scores from input features.
%
% Hyeonseob Nam, 2015
% 

n = size(ims,4);
nBatches = ceil(n/opts.batchSize);

net.layers = net.layers(1:2); % compute the relu of fc4
% add norm layer
%{
norm_layer.type = 'normalize';
norm_layer.name = 'norm3';
norm_layer.param = [5 2 1.000000000000000e-04 0.750000000000000];
net.layers = [net.layers, norm_layer];
%}

for i=1:nBatches
    
    batch = ims(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i));
    if(opts.useGpu)
        batch = gpuArray(batch);
    end
    
    res = vl_simplenn(net, batch, [], [], ...
        'disableDropout', true, ...
        'conserveMemory', true, ...
        'sync', true) ;
    
    f = gather(res(end).x) ;
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i)) = f;
    
end

% qyy normlization
x_max = max(max(feat));
x_min = min(min(feat));
feat = (feat-x_min)/(x_max-x_min);

end