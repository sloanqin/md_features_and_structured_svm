function [ res ] = st_svm_kernel_eval(x1, x2)
% st_svm_kernel_eval
% Main interface for structured svm
%
% INPUT:
%   st_svm  - parameters of structured svm: support patterns and support vectors
%   total_data  - fc4 features and example boxes
%   xi     - xi
%   y     - y
%
% OUTPUT:
%   st_svm - the st_svm has been updated
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

if (nargin<2)
	res = 1.0;
	return;
end

m_sigma = st_svm.kernerl_m_sigma;

res = exp(-m_sigma*squaredNorm(x1-x2));

end

function [res] = squaredNorm(x)
	res = sum(x.*x);
end