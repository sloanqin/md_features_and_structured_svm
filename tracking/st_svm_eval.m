function [ results ] = st_svm_eval(x_ind)
% st_svm_eval
% Main interface for structured svm
% eval the results for every examples in frame x_ind
%
% INPUT:
%   x_ind     - the index of frame
%
% OUTPUT:
%   results - the score of each examples in frame x_ind
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 
global total_data;

results = zeros(size(total_data(1,1,1,x_ind),4),1);
fvs = squeeze(total_data{1,1,1,x_ind});
y_rela = squeeze(total_data{1,1,3,x_ind});
for i=1:size(fvs,2)
	results(i) = st_svm_evaluate(fvs(:,i),y_rela(i,:));
end

end