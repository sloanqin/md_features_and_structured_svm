function st_svm_hnm( x_ind, opts )
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

% declare global variables
global total_data; 
global st_svm; 

targetLoc = total_data{:,:,2,x_ind}(1,:);
scores = st_svm_eval(x_ind);

for i=1:size(total_data{:,:,2,x_ind},1)
    rect_y1 = total_data{:,:,2,x_ind}(i,:);
    scores(i) =  scores(i) + loss(rect_y1, targetLoc);
end

[scores,idx] = sort(scores,'descend');
total_data{:,:,1,x_ind} = total_data{:,:,1,x_ind}(:,:,:, [1, idx(1: opts.svm_hnm_samples-1)]);
total_data{:,:,2,x_ind} = total_data{:,:,2,x_ind}([1, idx(1: opts.svm_hnm_samples-1)], :);
total_data{:,:,3,x_ind} = total_data{:,:,3,x_ind}([1, idx(1: opts.svm_hnm_samples-1)], :);

end

function [ re_loss ] = loss(rect_y1, rect_y2)
% loss
% compute loss of to rect
%
% INPUT:
%   rect_y1  - [left, top, width, height]
%
% OUTPUT:
%   re_loss - [0,1] = 1.0 - overlap_ratio
%
% Sloan Qin, 2017
% 
x1_min = rect_y1(1);
x1_max = rect_y1(1) + rect_y1(3);
y1_min = rect_y1(2);
y1_max = rect_y1(2) + rect_y1(4);
x2_min = rect_y2(1);
x2_max = rect_y2(1) + rect_y2(3);
y2_min = rect_y2(2);
y2_max = rect_y2(2) + rect_y2(4);

x0 = max(x1_min, x2_min);
x1 = min(x1_max, x2_max);
y0 = max(y1_min, y2_min);
y1 = min(y1_max, y2_max);

if (x0 >= x1 || y0 >= y1)
	re_loss = 1.0 - 0.0;
	return; 
end

areaInt = (x1-x0)*(y1-y0); % overlap area;

re_loss = 1.0 - areaInt/(double(rect_y1(3)*rect_y1(4))+double(rect_y2(3)*rect_y2(4))-areaInt);

end