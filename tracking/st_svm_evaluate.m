function [ f ] = st_svm_evaluate(x, y_rela)
% st_svm_evaluate
% Main interface for structured svm
%
% INPUT:
%   x     - feature vector
%   y_rela    - relative position [left-l_centre,top-t_centre,width-w_centre,height-h_centre]
%
% OUTPUT:
%   f - the score for input [x,yv]
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 
global total_data;

f = 0.0;
for i=1:size(st_svm.supportVectors,1)
	sv = st_svm.supportVectors{i,1};
	sv_x = squeeze(total_data{1,1,1,sv.x_ind}(:,:,:,sv.y_ind));
    % qyy debug
    if (size(x) ~= size(sv_x))
        fprintf('size(x) is %d\n',size(x));
        fprintf('size(sv_x) is %d\n',size(sv_x));
    end
	f = f + sv.b*st_svm_kernel_eval(x,sv_x); % betai*kernel(x,xi)
end

end