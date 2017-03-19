function st_svm_update(x_ind, whetherProcessNew)
% structured_svm_update
% Main interface for structured svm
%
% INPUT:
%   st_svm  - parameters of structured svm: support patterns and support vectors
%   total_data  - fc4 features and example boxes
%   ind     - The index of frame
%
% OUTPUT:
%   st_svm - the st_svm has been updated
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

% for debug
if(checkSVs()~=true)
    fprintf('wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
end

% qyy
if(whetherProcessNew)

    % new support pattern
    supportPattern.x_ind = x_ind;
    supportPattern.yi = 1;
    supportPattern.svCount = 0;% same with refCount in struck

    % add new support pattern to st_svm's support patterns
    st_svm.supportPatterns = [st_svm.supportPatterns; supportPattern];

    processNew(x_ind);
    budgetMaintenance();
end

for i = 1:10
	reprocess();
	budgetMaintenance();
end

end


function processNew(x_ind)
% processNew
% process a new frame, add new support vector and adjust beta and gradients
% the st_svm will be adjusted
%
% INPUT:
%   x_ind     - The index of frame
%
% Sloan Qin, 2017
% 

% declare global variables
global total_data;

% gradient is -f(x,y) since loss=0
xi = squeeze(total_data{1,1,1,x_ind}(:,:,:,1));
yi_rela = squeeze(total_data{1,1,3,x_ind}(:,:,:,1));
ip = addSupportVector(x_ind, 1, -st_svm_evaluate(xi, yi_rela));

[y_ind, min_grad] = minGradient(x_ind); % find min(gradient) of y
in = addSupportVector(x_ind, y_ind, min_grad);

SMOStep(ip, in); % adjust beta and gradient

end

function [ sv_ind ] = addSupportVector(x_ind, y_ind, grad)
% addSupportVector
% add a new support vector to st_svm
%
% INPUT:
%   x_ind  - The index of frame
%   y_ind     - The index of y for this sp
%   grad     - The gradient of this sv
%
% OUTPUT:
%   st_svm - the st_svm has been updated
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 
global total_data;

% new support vector
supportVector.b = 0;
supportVector.sp_ind = findSpind(x_ind);
supportVector.x_ind = x_ind;
supportVector.y_ind = y_ind;
supportVector.g = grad;

% add new support vector to st_svm's support vectors
st_svm.supportVectors = [st_svm.supportVectors; supportVector];

% refCount, svCount +1
st_svm.supportPatterns{supportVector.sp_ind,1}.svCount = st_svm.supportPatterns{supportVector.sp_ind,1}.svCount + 1;

% update kernel matrix
sv_ind = size(st_svm.supportVectors,1);
xi = squeeze(total_data{1,1,1,x_ind}(:,:,:,y_ind));
for i=1:sv_ind-1
	x_ind = st_svm.supportVectors{i, 1}.x_ind;
	y_ind = st_svm.supportVectors{i, 1}.y_ind;
	st_svm.m_k(i, sv_ind) = st_svm_kernel_eval(squeeze(total_data{1,1,1,x_ind}(:,:,:,y_ind)), xi);
	st_svm.m_k(sv_ind, i) = st_svm.m_k(i, sv_ind);
end
st_svm.m_k(sv_ind, sv_ind) = st_svm_kernel_eval(xi);

end


function [ sp_ind ] = findSpind(x_ind)
% findSpind
% find the index of sp for frame x_ind
%
% INPUT:
%   x_ind     - The index of frame
%
% OUTPUT:
%   sp_ind - the index of sp for frame x_ind
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm;

sp_ind = -1;
for i=1:size(st_svm.supportPatterns,1)
    if (st_svm.supportPatterns{i,1}.x_ind == x_ind)
		sp_ind = i;
		return;
    end
end
assert(sp_ind ~= -1);

end


function [ y_ind, min_grad ] = minGradient(x_ind)
% minGradient
% find the minium gradient of a frame
%
% INPUT:
%   x_ind     - The index of frame
%
% OUTPUT:
%   y_ind - the index of y for this frame(x_ind)
%   min_grad - the minium grad
%
% Sloan Qin, 2017
% 

% declare global variables
global total_data;

% res = [y_ind, min_grad]
y_ind = -1;
min_grad = realmax('double');

% all features and examples of this support pattern
xs = squeeze(total_data{1,1,1,x_ind});
ys = squeeze(total_data{1,1,2,x_ind});
y_relas = squeeze(total_data{1,1,3,x_ind});

% xi is at the first place
xi = xs(:,1);
yi = ys(1,:);

% traverse all x of this support pattern and compute grad
% find the minium grad
for i=1:size(xs,2)
	grad = -loss(yi,ys(i,:)) - st_svm_evaluate(xs(:,i),y_relas(i,:));
	if grad<min_grad
		min_grad = grad;
		y_ind = i;
	end
end

end


function [ y_ind, max_grad ] = maxGradient(x_ind)
% maxGradient
% find the maxium gradient of a frame
%
% INPUT:
%   x_ind     - The index of frame
%
% OUTPUT:
%   y_ind - the index of y for this frame(x_ind)
%   max_grad - the maxium grad
%
% Sloan Qin, 2017
% 

% declare global variables
global total_data;

% res = [y_ind, min_grad]
y_ind = -1;
max_grad = -realmax('double');

% all features and examples of this support pattern
xs = squeeze(total_data{1,1,1,x_ind});
ys = squeeze(total_data{1,1,2,x_ind});
y_relas = squeeze(total_data{1,1,3,x_ind});

% xi is at the first place
xi = xs(:,1);
yi = ys(1,:);

% traverse all x of this support pattern and compute grad
% find the minium grad
for i=1:size(xs,2)
	grad = -loss(yi,ys(i,:)) - st_svm_evaluate(xs(:,i),y_relas(i,:));
	if grad>max_grad
		max_grad = grad;
		y_ind = i;
	end
end

end

function SMOStep(sv_ipos, sv_ineg)
% SMOStep
% use SMO algorthim to update belta of svm
%
% INPUT:
%   sv_ipos  - index of positive support vector to update
%   sv_ineg  - index of negtive support vector to update
%
% OUTPUT:
%   st_svm - the st_svm will be updated
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

if (sv_ipos==sv_ineg)
	return;
end

% must be the same support pattern
sv_pos = st_svm.supportVectors{sv_ipos,1};
sv_neg = st_svm.supportVectors{sv_ineg,1};
assert(sv_pos.sp_ind == sv_neg.sp_ind);

if ( (sv_pos.g-sv_neg.g) < 1e-5 )
	% fprintf('skipping SMO')
else
	kii = st_svm.m_k(sv_ipos, sv_ipos) + st_svm.m_k(sv_ineg, sv_ineg) - 2*st_svm.m_k(sv_ipos, sv_ineg);
	lu = (sv_pos.g-sv_neg.g)/kii;
	% no need to clamp against 0 since we'd have skipped in that case
	% yi_ind == 1
	l = min(lu, st_svm.svmC*double(sv_pos.y_ind == 1) - sv_pos.b);

	st_svm.supportVectors{sv_ipos,1}.b = st_svm.supportVectors{sv_ipos,1}.b + l;
	st_svm.supportVectors{sv_ineg,1}.b = st_svm.supportVectors{sv_ineg,1}.b - l;

	% update gradients
	for i=1:size(st_svm.supportVectors,1)
		st_svm.supportVectors{i,1}.g = st_svm.supportVectors{i,1}.g - l*(st_svm.m_k(i, sv_ipos) - st_svm.m_k(i, sv_ineg));
	end
end

% for debug
%fprintf('show1,sv_ipos:%d,sv_ineg:%d\n',sv_ipos,sv_ineg);
%fprintf('size(st_svm.supportVectors,1):%d\n',size(st_svm.supportVectors,1));
%fprintf('beta of sv_ipos:%.22f,sv_ineg:%.22f\n',st_svm.supportVectors{sv_ipos,1}.b,st_svm.supportVectors{sv_ineg,1}.b);

% do not remove first sp
if(countSVNumOfSameSP(sv_ipos)==2 && st_svm.supportVectors{sv_ipos,1}.x_ind==1)
    return;
end

% add by qyy,when a sp has only 2 svs and one of them less then 1e-8,delete
% both of them
bothDelete = (countSVNumOfSameSP(sv_ipos)<=2 && (abs(st_svm.supportVectors{sv_ipos,1}.b) < 1e-8 || abs(st_svm.supportVectors{sv_ineg,1}.b) < 1e-8));

% check if we should remove either sv now
deleteIPos = (countSVNumOfSameSP(sv_ipos)>=3 && (abs(st_svm.supportVectors{sv_ipos,1}.b) < 1e-8) && (sv_ipos ~= findPosBetaSVOfSameSP(sv_ipos)));
if ( deleteIPos || bothDelete)
	%fprintf('removeSupportVector sv_ipos in function SMOStep,beta>0:%d, beta is %.22f\n',st_svm.supportVectors{sv_ipos,1}.b>0,st_svm.supportVectors{sv_ipos,1}.b);
    %if (st_svm.supportVectors{sv_ipos,1}.b>0)
        %fprintf('!!!!!!!!!!!!!!!!!!supportVectors{sv_ind,1}.b>0 !!!!!!!!!!!!\n');
    %end
	%fprintf('same SP with sv_ipos is %d\n',debugCountSVNumOfSameSP(sv_ipos));
	removeSupportVector(sv_ipos);
	if ( sv_ineg == (uint32(size(st_svm.supportVectors,1)) + 1) )
		% ineg and ipos will have been swapped during sv removal
		sv_ineg = sv_ipos;
	end	
end

% qyy debug
%fprintf('show2,sv_ipos:%d,sv_ineg:%d\n',sv_ipos,sv_ineg);
%fprintf('size(st_svm.supportVectors,1)=%d\n',size(st_svm.supportVectors,1));
%fprintf('sv_ineg=%d\n',sv_ineg);
deleteINeg = (countSVNumOfSameSP(sv_ineg)>=3 && (abs(st_svm.supportVectors{sv_ineg,1}.b) < 1e-8) && (sv_ineg ~= findPosBetaSVOfSameSP(sv_ineg)));
if ( deleteINeg || bothDelete)
	%fprintf('removeSupportVector sv_ineg in function SMOStep,beta>0:%d,beta is %.22f\n',st_svm.supportVectors{sv_ineg,1}.b>0,st_svm.supportVectors{sv_ineg,1}.b);
    %fprintf('same SP with sv_ineg is %d\n',debugCountSVNumOfSameSP(sv_ineg));
  	removeSupportVector(sv_ineg);
end

end

function [ pos_sv_ind ] = findPosBetaSVOfSameSP(sv_ind)
% findPosBetaSVOfSameSP
% find postive beta supportVector of the same supportPattern with sv_ind
%
% INPUT:
%   sv_ind  - index of supportVector
%
% OUTPUT:
%   num - index of supportVector with beta>0
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 
pos_sv_ind = -1;
for k=1:uint32(size(st_svm.supportVectors,1))
	if ((st_svm.supportVectors{k,1}.x_ind == st_svm.supportVectors{sv_ind,1}.x_ind) && (st_svm.supportVectors{k,1}.y_ind == 1))
		pos_sv_ind = k;
	end
end
assert(pos_sv_ind ~= -1);
return;
end

function [ num ] = countSVNumOfSameSP(sv_ind)
% countSVNumOfSameSP
% count the num of supportVectors for the same supportPattern with supportVector sv_ind
%
% INPUT:
%   sv_ind  - index of supportVector
%
% OUTPUT:
%   num - number of supportVectors,>=1
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 
num = 0;

for k=1:uint32(size(st_svm.supportVectors,1))
	if (st_svm.supportVectors{k,1}.x_ind == st_svm.supportVectors{sv_ind,1}.x_ind)
		num = num + 1;
	end
end
return;
end

function [ num ] = debugCountSVNumOfSameSP(sv_ind)
% countSVNumOfSameSP
% count the num of supportVectors for the same supportPattern with supportVector sv_ind
% this func is designed for debug
%
% INPUT:
%   sv_ind  - index of supportVector
%
% OUTPUT:
%   num - number of supportVectors,>=1
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 
num = 0;

fprintf('in func countSVNumOfSameSP,sv_ind is %d,beta is %f, all the sv in same sp is...\n',sv_ind,st_svm.supportVectors{sv_ind,1}.b);
for k=1:uint32(size(st_svm.supportVectors,1))
	if (st_svm.supportVectors{k,1}.x_ind == st_svm.supportVectors{sv_ind,1}.x_ind)
		num = num + 1;
		fprintf('sv_ind is %d,beta is %f,',k,st_svm.supportVectors{k,1}.b);
	end
end
fprintf('......\n');
return;
end

function [ checkAns ] = checkSVs()
% checkSVs
% check whether every sv has a pos sv of same sp
% this func is designed for debug
%
% OUTPUT:
%   true/false - right/wrong
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

for i=1:uint32(size(st_svm.supportVectors,1))
	if (st_svm.supportVectors{i,1}.b < 0.0)
		% find corresponding positive sv
		j = -1;
        for k=1:uint32(size(st_svm.supportVectors,1))
            if (st_svm.supportVectors{k,1}.y_ind == 1 && st_svm.supportVectors{k,1}.x_ind == st_svm.supportVectors{i,1}.x_ind)
				j = k;
				break;
            end
        end
        
        if(j==-1)
            checkAns = false;
            return;
        end
	end
end

checkAns = true;
return;
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


function budgetMaintenance()
% budgetMaintenance
% limit number of sv
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

if st_svm.svmBudgetSize>0
	while (uint32(size(st_svm.supportVectors,1)) > st_svm.svmBudgetSize)
		budgetMaintenanceRemove();
	end
end

end

function budgetMaintenanceRemove()
% budgetMaintenanceRemove
% remove support vector
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 
global total_data;

% find negative sv with smallest effect on discriminant function if removed
minVal = realmax('double');
in = -1;
ip = -1;

for i=1:uint32(size(st_svm.supportVectors,1))
	if (st_svm.supportVectors{i,1}.b < 0.0)
		% find corresponding positive sv
		j = -1;
        for k=1:uint32(size(st_svm.supportVectors,1))
            % qyy,st_svm.supportVectors{k,1}.y_ind==1 is better than st_svm.supportVectors{k,1}.b > 0.0
            if (st_svm.supportVectors{k,1}.y_ind==1 && st_svm.supportVectors{k,1}.x_ind == st_svm.supportVectors{i,1}.x_ind)
				j = k;
				break;
            end
        end
        %for debug
        %fprintf('st_svm.supportVectors size is %d,%d\n',size(st_svm.supportVectors));
        %fprintf('st_svm.m_k size is %d,%d\n',size(st_svm.m_k));
        %fprintf('i and j is %d, %d\n',i,j);
		val = ((st_svm.supportVectors{i,1}.b)^2) * (st_svm.m_k(i,i) + st_svm.m_k(j,j) - 2.0*st_svm.m_k(i,j));
		if (val < minVal)
			minVal = val;
			in = i;
			ip = j;
		end
	end
end

% adjust weight of positive sv to compensate for removal of negative
st_svm.supportVectors{ip,1}.b = st_svm.supportVectors{ip,1}.b + st_svm.supportVectors{in,1}.b;

% remove negative sv
removeSupportVector(in);
if (ip == (uint32(size(st_svm.supportVectors,1))+1) )
		% ip and in will have been swapped during support vector removal
		ip = in;
end
	
% qyy modified
if (countSVNumOfSameSP(ip)==1)
	% also remove positive sv
	removeSupportVector(ip);
end

% update gradients
% TODO: this could be made cheaper by just adjusting incrementally rather than recomputing
for i=1:uint32(size(st_svm.supportVectors,1))
	x_ind = st_svm.supportVectors{i,1}.x_ind;
	y_ind = st_svm.supportVectors{i,1}.y_ind;
	sp_ind = st_svm.supportVectors{i,1}.sp_ind;
	yi = squeeze(total_data{1,1,2,x_ind}(1,:));
	y = squeeze(total_data{1,1,2,x_ind}(y_ind,:));
	y_rela = squeeze(total_data{1,1,3,x_ind}(y_ind,:));
	x = squeeze(total_data{1,1,1,x_ind}(:,:,:,y_ind));
	st_svm.supportVectors{i,1}.g = -loss(y,yi) - st_svm_evaluate(x,y_rela);
end

end


function reprocess()
% reprocess
% processOld and optimize
%
% Sloan Qin, 2017
% 

processOld();
for i=1:10
	optimize();
end

end

function processOldBackup()
% processOld
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

if (size(st_svm.supportPatterns,1) == 0) 
    return; 
end

% qyy,choose frame to process
x_ind = uint32(rand()*st_svm.x_ind);
if (x_ind==0)
	x_ind = 1;
end

if(st_svm.targetScores(x_ind,1)<0)
    return;
end

% qyy, find potentially new sv with smallest grad
addSupportPattern = true;
ip = -1;
[ yp_ind, max_grad ] = maxGradient(x_ind);
for i=1:size(st_svm.supportVectors,1)
	sv = st_svm.supportVectors{i,1};
	if (sv.x_ind ~= x_ind) 
		continue;
    end
    addSupportPattern = false;
	if (sv.y_ind == yp_ind)
		ip = i;
		break;
	end
end

if(addSupportPattern)
    fprintf('addSupportPattern...\n');
    % new support pattern
    supportPattern.x_ind = x_ind;
    supportPattern.yi = 1;
    supportPattern.svCount = 0;% same with refCount in struck

    % add new support pattern to st_svm's support patterns
    st_svm.supportPatterns = [st_svm.supportPatterns; supportPattern];
end

if (ip == -1)
	ip = addSupportVector(x_ind, yp_ind, max_grad);
end

% find potentially new sv with smallest grad
[ yn_ind, min_grad ] = minGradient(x_ind);
in = -1;
for i=1:size(st_svm.supportVectors,1)
	sv = st_svm.supportVectors{i,1};
	if (sv.x_ind ~= x_ind) 
		continue;
	end
	if (sv.y_ind == yn_ind)
		in = i;
		break;
	end
end

if (in == -1)
	in = addSupportVector(x_ind, yn_ind, min_grad);
end

fprintf('in func processOld x_ind:%d,yp_ind:%d,yn_ind:%d,ip:%d,in:%d\n',x_ind,yp_ind,yn_ind,ip,in);
SMOStep(ip, in);

end

function processOld()
% processOld
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

if (size(st_svm.supportPatterns,1) == 0) 
    return; 
end

% choose pattern to process
ind = uint32(rand()*size(st_svm.supportPatterns,1));
if (ind==0)
	ind = 1;
end

% find existing sv with largest grad and nonzero beta
ip = -1;
maxGrad = -realmax('double');
for i=1:size(st_svm.supportVectors,1)
	if (st_svm.supportVectors{i,1}.sp_ind ~= ind) 
		continue;
	end
	sv = st_svm.supportVectors{i,1};
	if (sv.g>maxGrad && sv.b <= st_svm.svmC*double(sv.y_ind == 1))%qyy modified
		ip = i;
		maxGrad = sv.g;
	end
end

assert(ip ~= -1);
if (ip == -1) 
	return;
end

% find potentially new sv with smallest grad
[ y_ind, min_grad ] = minGradient(st_svm.supportPatterns{ind,1}.x_ind);
in = -1;
for i=1:size(st_svm.supportVectors,1)
	sv = st_svm.supportVectors{i,1};
	if (sv.sp_ind ~= ind) 
		continue;
	end
	if (sv.y_ind == y_ind)
		in = i;
		break;
	end
end

if (in == -1)
	in = addSupportVector(st_svm.supportPatterns{ind,1}.x_ind, y_ind, min_grad);
end

SMOStep(ip, in);

end

function optimize()
% optimize
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

if (size(st_svm.supportPatterns,1) == 0) 
    return; 
end
	
% choose pattern to optimize
ind = uint32(rand()*size(st_svm.supportPatterns,1));
if (ind==0)
	ind = 1;
end

ip = -1;
in = -1;
maxGrad = -realmax('double');
minGrad = realmax('double');
for i=1:size(st_svm.supportVectors,1) 
	sv = st_svm.supportVectors{i,1};
    if (sv.sp_ind ~= ind) 
		continue;
    end
	%if (sv.g>maxGrad && sv.b<st_svm.svmC*double(sv.y_ind==1))
    if (sv.g>maxGrad && sv.b<=st_svm.svmC*double(sv.y_ind==1))%qyy modified
		ip = i;
		maxGrad = sv.g;
	end
	if (sv.g<minGrad)
		in = i;
		minGrad = sv.g;
	end
end

%debug
%fprintf('in func optimize,num of sv for same sp is %d,ip is %d,in is %d\n',debugCountSVNumOfSameSP(tempSvInd),ip,in);

if(ip == -1 || in == -1)
    fprintf('in func optimize ip:%d,in:%d\n',ip,in);
end
assert(ip ~= -1 && in ~= -1);
if (ip == -1 || in == -1)
	% this shouldn't happen
	sprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	return;
end

SMOStep(ip, in);

end


function removeSupportVector(sv_ind)
% removeSupportVector
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

fprintf('removeSupportVector %d\n',sv_ind);

sv = st_svm.supportVectors{sv_ind,1};
st_svm.supportPatterns{sv.sp_ind,1}.svCount = st_svm.supportPatterns{sv.sp_ind,1}.svCount - 1;
if (st_svm.supportPatterns{sv.sp_ind,1}.svCount == 0)
	% also remove the support pattern
    fprintf('removeSupportPattern %d\n',sv.sp_ind);
	st_svm.supportPatterns = [st_svm.supportPatterns(1:sv.sp_ind-1);st_svm.supportPatterns(sv.sp_ind+1:end)];
	% update sp_ind of supportVectors
	for i=1:size(st_svm.supportVectors,1)
		if (st_svm.supportVectors{i,1}.sp_ind>sv.sp_ind)
			st_svm.supportVectors{i,1}.sp_ind = st_svm.supportVectors{i,1}.sp_ind - 1;
		end
	end
end

% make sure the support vector is at the back, this
% lets us keep the kernel matrix cached and valid
if (sv_ind < size(st_svm.supportVectors,1))
	swapSupportVectors(sv_ind, size(st_svm.supportVectors,1));
	sv_ind = size(st_svm.supportVectors,1);
end

% delete support vector	
st_svm.supportVectors = st_svm.supportVectors(1:end-1,1);

end


function swapSupportVectors(sv_ind1, sv_ind2)
% swapSupportVectors
%
% Sloan Qin, 2017
% 

% declare global variables
global st_svm; 

tmp = st_svm.supportVectors{sv_ind1,1};
st_svm.supportVectors{sv_ind1,1} = st_svm.supportVectors{sv_ind2,1};
st_svm.supportVectors{sv_ind2,1} = tmp;

row1 = st_svm.m_k(sv_ind1,:);
st_svm.m_k(sv_ind1,:) = st_svm.m_k(sv_ind2,:);
st_svm.m_k(sv_ind2,:) = row1;

col1 = st_svm.m_k(:,sv_ind1);
st_svm.m_k(:,sv_ind1) = st_svm.m_k(:,sv_ind2);
st_svm.m_k(:,sv_ind2) = col1;

end