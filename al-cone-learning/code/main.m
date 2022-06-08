% This code is in parts an adaption of https://github.com/pujols/Zero-shot-learning-journal
%% Specify parameters

% Allows for using arrays for tuning hyperparameters
optSVR.C_SVR_a =32;
optSVR.nu_a = 0.0165;
optSVR.gamma_a = 2;
optSVR.pca_d_a = [500,1000];
optSVM.C_SVM_a = [1,3,5,7];
optSVM.SVM_Kernel = {{'polynomial',5},{'polynomial',7},{'polynomial',9}};
optSVM.Kernel_param = [1,3,5,7];

% train or test
value = 'train';

%% read data
% five-fold cross validation
nr_fold = 5;

addpath('../tool/libsvm-3.25/matlab');
addpath('../comp/misc')
addpath('../comp/EXEM/codes')


%% 
dataset = 'AWA2_SS';

%% Parameters setting
feature_type = 'resnet';

opt.ind_split = [];

% Xtr: Features for training
% Ytr: Labels for training
% Xte: Features for test
% Yte: Labels for test

% Dataloader loads data and normalizes them
[X_train, Y_train, X_test, Y_test, ~, class_order] = data_loader(dataset, opt, feature_type, 'not');

%% Cross-Validation
attr_space_bin = binary_attr_loader(dataset);

% find labels used in training, remove the ones used only for testing from
% the attribute space and renumber the training labels starting with 1
if strcmp(value, 'train')
    fold_loc = cv_split('train', Y_train, class_order, nr_fold);
    u=unique(Y_train);
    attr_space_bin = attr_space_bin(u, :);

    for i=1:size(Y_train,1)
        Y_train(i) = find(Y_train(i)==u);
    end
else 
    fold_loc = {};
    nr_fold = 1;
end


% Return of normed version of attribute distribution
attr_space = attr_space_bin;
attr_space(attr_space==0)=-1;
norm_method = 'L2';
attr_space = get_class_signatures(attr_space, norm_method);

val_acc = zeros(length(optSVR.C_SVR_a), length(optSVR.nu_a), length(optSVR.gamma_a), length(optSVM.C_SVM_a), length(optSVM.Kernel_param));
val_hits_at_1 = zeros(length(optSVR.C_SVR_a), length(optSVR.nu_a), length(optSVR.gamma_a), length(optSVM.C_SVM_a), length(optSVM.Kernel_param));
val_avg_answer_1 = zeros(length(optSVR.C_SVR_a), length(optSVR.nu_a), length(optSVR.gamma_a), length(optSVM.C_SVM_a), length(optSVM.Kernel_param));
val_avg_answer = zeros(length(optSVR.C_SVR_a), length(optSVR.nu_a), length(optSVR.gamma_a), length(optSVM.C_SVM_a), length(optSVM.Kernel_param));
val_prec = zeros(length(optSVR.C_SVR_a), length(optSVR.nu_a), length(optSVR.gamma_a), length(optSVM.C_SVM_a), length(optSVM.Kernel_param));

prec_rec  = {};
avg_hits_at ={};


for i_1=1:size(optSVR.pca_d_a,2)
    pca_d = optSVR.pca_d_a(i_1);

    % PCA Projection
    [mean_Xtr_PCA, V] = do_pca(X_train);
    X_train = bsxfun(@minus, X_train, mean_Xtr_PCA);
    X_test = bsxfun(@minus, X_test, mean_Xtr_PCA);
    
    % PCA projection
    mapped_Xtr = X_train * V(:, 1 : pca_d);
    mapped_Xte = X_test * V(:, 1 : pca_d);
    
    L_i_2 = length(optSVR.C_SVR_a);
    L_i_3 = length(optSVR.nu_a);
    L_i_4 = size(optSVR.gamma_a,2);
    L_i_5 = size(optSVM.C_SVM_a,2);
    L_i_6 = length(optSVM.SVM_Kernel);
    L_i_8 = length(optSVM.Kernel_param);

    %% Compute
    % Test of parameters
    for i_2=1:L_i_2
        for i_3=1:L_i_3
            for i_4=1:L_i_4
                % cross validation
                for f = 1:nr_fold
                    % parameter assignment
                    gamma = optSVR.gamma_a(i_4);
                    direct_test = [optSVR.C_SVR_a(i_2), optSVR.nu_a(i_3), gamma, pca_d];

                    if strcmp(value, 'train')
                        % Features
                        X_train_f = mapped_Xtr;
                        X_train_f(fold_loc{f}, :) = [];
                        Y_train_f = Y_train;
                        Y_train_f(fold_loc{f}) = [];
                        % Validation
                        X_val_f = mapped_Xtr(fold_loc{f}, :);
                        Y_val_f = Y_train(fold_loc{f});
                                           
                    else
                        % Features
                        X_train_f = mapped_Xtr;
                        Y_train_f = Y_train;
                        % Test
                        X_val_f = mapped_Xte;
                        Y_val_f = Y_test;
                    
                    end
                    % Creation of the geometric model (using max-matching hamming distance as base)
                    % row in geometric model corresponds to number of label
                    attr_space_val = attr_space(unique(Y_val_f),:);
                    attr_space_bin_val = attr_space_bin(unique(Y_val_f),:);
                    geom_model = build_geometric_model(attr_space_val);

                
                    % 1. Calculation of regressors for mapping from attribute to feature space
                    regressors = EXEM_simplified(X_train_f, Y_train_f, attr_space,  direct_test);
                    
                    [~,std_X] = compute_class_stat(Y_train_f, X_train_f);
                    avg_std_Xtr = mean(std_X, 1);

                    for i_5=1:L_i_5
                        for i_6=1:L_i_6
                            for i_8=1:L_i_8
                     
                                C_SVM = optSVM.C_SVM_a(i_5);
                                SVM_Kernel = optSVM.SVM_Kernel{i_6};
    
                                % Calculation of visual exemplars for related classes and calculate SVM
                                % for each dimension separately
                                [SVM] =  classify_dimension(regressors, attr_space_val, attr_space_bin_val, geom_model, attr_space(unique(Y_train_f),:), gamma, C_SVM, SVM_Kernel, optSVM.Kernel_param(i_8));
                                                          
                            
                                %% 3. Evaluation
                                % adapting numbers for correct
                                % classification
                                u= unique(Y_val_f);
                                for l=1:size(Y_val_f,1)
                                    Y_val_f(l) = find(Y_val_f(l)==u);
                                end
                                [acc,prec, avg_answer, hits_at_1,avg_answer_1, prec_reca] = evaluate(X_val_f,Y_val_f, geom_model, SVM);
                                
                                prec_rec{f} = prec_reca;

                                val_acc(i_2,i_3,i_4,i_5,i_8,:) = val_acc(i_2,i_3,i_4,i_5,i_8)+acc/nr_fold;
                                val_hits_at_1(i_2,i_3,i_4,i_5,i_8,:) = val_hits_at_1(i_2,i_3,i_4,i_5,i_8)+hits_at_1/nr_fold;
                                val_avg_answer(i_2,i_3,i_4,i_5,i_8,:) = val_avg_answer(i_2,i_3,i_4,i_5,i_8)+avg_answer/nr_fold;
                                val_avg_answer_1(i_2,i_3,i_4,i_5,i_8,:) = val_avg_answer_1(i_2,i_3,i_4,i_5,i_8)+avg_answer_1/nr_fold;
                                val_prec(i_2,i_3,i_4,i_5,i_8,:) = val_prec(i_2,i_3,i_4,i_5,i_8)+prec/nr_fold;
                            end
                        end
                    end
                end
            end
        end
    end
end
save(['../results/' dataset '_' value '_' char(datetime('now')) '_split.mat'],...
    'val_acc','val_hits_at_1','val_avg_answer','val_avg_answer_1','val_prec','optSVM','optSVR',"SVM_Kernel");
                        
                    