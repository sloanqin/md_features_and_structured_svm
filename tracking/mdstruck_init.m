function mdstruck_init()
% MDSTRUCK_INIT
% Initialize structured svm.
%
% sloan qin, 2017
% 

global st_svm;

% for support patterns
st_svm.supportPatterns = cell(0,1);

% for support vectors
st_svm.supportVectors = cell(0,1);

% max number of support vectors
st_svm.kMaxSVs = 2000;

% svmBudgetSize, limit the number of support vectors
st_svm.svmBudgetSize = 3;

% kernel matrix
if st_svm.svmBudgetSize>0
	st_svm.N = st_svm.svmBudgetSize;
else
	st_svm.N = st_svm.kMaxSVs;
end
st_svm.m_k = zeros(st_svm.N,st_svm.N);

% kernerl kind
st_svm.kernerl = 'GaussianKernel'; % LinearKernel,GaussianKernel etc

% GaussianKernel kernel-params
st_svm.kernerl_m_sigma = 0.2;

% SVM regularization parameter.
st_svm.svmC = 100.0;

end