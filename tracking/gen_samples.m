function [ bb_samples ] = gen_samples(type, bb, n, opts, trans_f, scale_f)
% GEN_SAMPLES
% Generate sample bounding boxes.
%
% TYPE: sampling method
%   'gaussian'          generate samples from a Gaussian distribution centered at bb
%                       -> positive samples, target candidates                        
%   'uniform'           generate samples from a uniform distribution around bb
%                       -> negative samples
%   'uniform_aspect'    generate samples from a uniform distribution around bb with varying aspect ratios
%                       -> training samples for bbox regression
%   'whole'             generate samples from the whole image
%                       -> negative samples at the initial frame
%
% Hyeonseob Nam, 2015
% 

h = opts.imgSize(1); w = opts.imgSize(2);

% [center_x center_y width height]
sample = [bb(1)+bb(3)/2 bb(2)+bb(4)/2, bb(3:4)];
samples = repmat(sample, [n, 1]);

switch (type)
    case 'gaussian'
        samples(:,1:2) = samples(:,1:2) + trans_f * round(mean(bb(3:4))) * max(-1,min(1,0.5*randn(n,2)));
        samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*max(-1,min(1,0.5*randn(n,1)))),1,2);
    case 'uniform'
        samples(:,1:2) = samples(:,1:2) + trans_f * round(mean(bb(3:4))) * (rand(n,2)*2-1);
        samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*(rand(n,1)*2-1)),1,2);
    case 'uniform_aspect'
        samples(:,1:2) = samples(:,1:2) + trans_f * repmat(bb(3:4),n,1) .* (rand(n,2)*2-1);
        samples(:,3:4) = samples(:,3:4) .* opts.scale_factor.^(rand(n,2)*4-2);
        samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*rand(n,1)),1,2);
    case 'radial' % qyy add
		rstep = double(opts.svm_update_radius)/opts.svm_nr;
		tstep = 2*pi/opts.svm_nt;
		radius = [rstep:rstep:opts.svm_update_radius];
		angle = [0:tstep:2*pi-0.0000001] + repmat([tstep/2,0],[1,opts.svm_nt/2]);
		dx = [0;reshape(radius' * cos(angle),[],1)];
		dy = [0;reshape(radius' * sin(angle),[],1)];
        samples(:,1) = samples(:,1) + dx;
        samples(:,2) = samples(:,2) + dy;
        %samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*max(-1,min(1,0.5*randn(n,1)))),1,2);
    case 'pixel' % qyy add
		dx = [-opts.svm_eval_radius:1:opts.svm_eval_radius];
		dy = [-opts.svm_eval_radius:1:opts.svm_eval_radius];
		dx(find(dx==0)) = dx(1); dx(1) = 0; % replace 0 to first place
		dy(find(dy==0)) = dy(1); dy(1) = 0;
		dx = reshape(repmat(dx,[1,size(dy,2)]),[],1);
		dy = reshape(repmat(dy,[size(dy,2),1]),[],1);
        index =  find((dx(:).*dx(:) + dy(:).*dy(:)) <= opts.svm_eval_radius^2);
        dx = dx(index);
        dy = dy(index);
        samples(:,1) = samples(:,1) + dx;
        samples(:,2) = samples(:,2) + dy;
        %samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*(rand(n,1)*2-1)),1,2);
    case 'whole'
        range = round([bb(3)/2 bb(4)/2 w-bb(3)/2 h-bb(4)/2]);
        stride = round([bb(3)/5 bb(4)/5]);
        [dx, dy, ds] = meshgrid(range(1):stride(1):range(3), range(2):stride(2):range(4), -5:5);
        windows = [dx(:) dy(:) bb(3)*opts.scale_factor.^ds(:) bb(4)*opts.scale_factor.^ds(:)];
        
        samples = [];
        while(size(samples,1)<n)
            samples = cat(1,samples,...
                windows(randsample(size(windows,1),min(size(windows,1),n-size(samples,1))),:));
        end
end

% width and height are limited in [10,w-10],[10,h-10]
samples(:,3) = max(10,min(w-10,samples(:,3)));
samples(:,4) = max(10,min(h-10,samples(:,4)));

% [left top width height]
% limit left in [-width/2,w-width/2],limit top in [-height/2,h-height/2],ie the center is in the image
bb_samples = [samples(:,1)-samples(:,3)/2, samples(:,2)-samples(:,4)/2, samples(:,3:4)];
bb_samples(:,1) = max(1-bb_samples(:,3)/2,min(w-bb_samples(:,3)/2, bb_samples(:,1)));
bb_samples(:,2) = max(1-bb_samples(:,4)/2,min(h-bb_samples(:,4)/2, bb_samples(:,2)));
bb_samples = round(bb_samples);


end
