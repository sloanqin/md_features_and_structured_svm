function [ result ] = mdstruck_run(images, region, net, display)
% MDNET_RUN
% Main interface for MDNet tracker
%
% INPUT:
%   images  - 1xN cell of the paths to image sequences
%   region  - 1x4 vector of the initial bounding box [left,top,width,height]
%   net     - The path to a trained MDNet
%   display - True for displying the tracking result
%
% OUTPUT:
%   result - Nx4 matrix of the tracking result Nx[left,top,width,height]
%
% Hyeonseob Nam, 2015
% 

if(nargin<4), display = true; end

% declare global variables
global st_svm; 
global total_data;

%% Initialization
fprintf('Initialization...\n');

nFrames = length(images);

img = imread(images{1});
if(size(img,3)==1), img = cat(3,img,img,img); end
targetLoc = region;
result = zeros(nFrames, 4); result(1,:) = targetLoc;

[net_conv, net_fc, opts] = mdnet_init(img, net);

%% Train a bbox regressor
if(opts.bbreg)
    fprintf('  bbox regressor\n');
    pos_examples = gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
    r = overlap_ratio(pos_examples,targetLoc);
    pos_examples = pos_examples(r>0.6,:);
    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
    feat_conv = mdnet_features_convX(net_conv, img, pos_examples, opts);
    
    X = permute(gather(feat_conv),[4,3,1,2]);
    X = X(:,:);
    bbox = pos_examples;
    bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
    bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
end

%% Extract training examples
fprintf('  extract features...\n');

% draw positive/negative samples
pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_init*2, opts, 0.1, 5);
r = overlap_ratio(pos_examples,targetLoc);
pos_examples = pos_examples(r>opts.posThr_init,:);
pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

neg_examples = [gen_samples('uniform', targetLoc, opts.nNeg_init, opts, 1, 10);...
    gen_samples('whole', targetLoc, opts.nNeg_init, opts)];
r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);

examples = [pos_examples; neg_examples];
pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

% extract conv3 features
feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
pos_data = feat_conv(:,:,:,pos_idx);
neg_data = feat_conv(:,:,:,neg_idx);


%% Learning CNN
fprintf('  training cnn...\n');
net_fc = mdnet_finetune_hnm(net_fc,pos_data,neg_data,opts,...
    'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);

%% Initialize displayots
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');
    
    hd = imshow(img,'initialmagnification','fit'); hold on;
    rectangle('Position', targetLoc, 'EdgeColor', [1 0 0], 'Linewidth', 3);
    set(gca,'position',[0 0 1 1]);
    
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
end

%% Prepare training data for structured svm online update
%% total_data(1,1,1,:):features,total_data(1,1,2,:):y,total_data(1,1,3,:):yv
total_data = cell(1,1,3,nFrames);

examples = gen_samples('radial', targetLoc, opts.svm_samples, opts, 2, 5);

feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
feat_fc4 = mdnet_features_fc4(net_fc, feat_conv, opts);
total_data{:,:,1,1} = feat_fc4(:,:,:,:);
total_data{:,:,2,1} = examples;
total_data{:,:,3,1} = examples - repmat(targetLoc,[size(examples,1),1]);

%% for debug
%{
y = importdata('./dataset/debug/y.txt',',');
y_rela = importdata('./dataset/debug/yrela.txt',',');
feat = importdata('./dataset/debug/feat.txt',',');
feat = feat';
total_data{:,:,1,1} = reshape(feat,[1,1,192,81]);
total_data{:,:,2,1} = y;
total_data{:,:,3,1} = y_rela;
%}

%% st_svm initialise
mdstruck_init();

%% structured svm update
st_svm_update(1);

success_frames = 1;
trans_f = opts.trans_f;	
scale_f = opts.scale_f;

%% Main loop
for To = 2:nFrames;
    fprintf('Processing frame %d/%d... \n', To, nFrames);
	fprintf('supportPatterns/supportVectors is %d/%d... \n', size(st_svm.supportPatterns,1), size(st_svm.supportVectors,1));
    
    img = imread(images{To});
    if(size(img,3)==1), img = cat(3,img,img,img); end
    
    spf = tic;
    %% Estimation
    % draw target candidates
    examples = gen_samples('pixel', targetLoc, opts.svm_eval_samples, opts, trans_f, scale_f);
    feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
	feat_fc4 = mdnet_features_fc4(net_fc, feat_conv, opts);
    total_data{:,:,1,To} = feat_fc4(:,:,:,:);
    total_data{:,:,2,To} = examples;
    total_data{:,:,3,To} = examples - repmat(targetLoc,[size(examples,1),1]); 
   
    % evaluate the candidates
    scores = st_svm_eval(To);
    [scores,idx] = sort(scores,'descend');
    target_score = scores(1,1);
    targetLoc = examples(idx(1,1),:);
    
    % final target
    result(To,:) = targetLoc;
    
    % extend search space in case of failure
    if(target_score<0)
        trans_f = min(1.5, 1.1*trans_f);
    else
        trans_f = opts.trans_f;
    end
    
    % bbox regression
    if(opts.bbreg && target_score>0)
        X_ = permute(gather(feat_conv(:,:,:,idx(1:5))),[4,3,1,2]);
        X_ = X_(:,:);
        bbox_ = examples(idx(1:5),:);
        pred_boxes = predict_bbox_regressor(bbox_reg.model, X_, bbox_);
        result(To,:) = round(mean(pred_boxes,1));
    end
    
    %% Prepare training data
    %if(target_score>0)
        examples = gen_samples('radial', targetLoc, opts.svm_samples, opts, 0.1, 5);

		feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
		feat_fc4 = mdnet_features_fc4(net_fc, feat_conv, opts);
		total_data{:,:,1,To} = feat_fc4(:,:,:,:);
		total_data{:,:,2,To} = examples;
        total_data{:,:,3,To} = examples - repmat(targetLoc,[size(examples,1),1]); 
        
        success_frames = [success_frames, To];
    %end
    
    %% structured svm update
	st_svm_update(To);
    
    spf = toc(spf);
    fprintf('%f seconds\n',spf);
    
    %% Display
    if display
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',img); hold on;
        
        rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
        set(gca,'position',[0 0 1 1]);
        
        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        drawnow;
    end
end









