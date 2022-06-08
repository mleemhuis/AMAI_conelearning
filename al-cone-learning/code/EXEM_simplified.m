% Simplified version of EXEM; based on https://github.com/pujols/Zero-shot-learning-journal

function [regressors] = EXEM_simplified(X_train, Y_train, attr_space,  direct_test)

%% Input
% opt: opt.C: the regularizer coefficient of nu-SVR (e.g, 2 .^ 0)
%      opt.nu: the nu-SVR parameter (e.g, 2 .^ (-10 : 0))
%      opt.gamma: the RBF scale parameter (e.g., 2 .^ (-5 : 5))
%      opt.pca_d: the PCA dimensionality (e.g., 500)
% direct_test: test on a specific [C, nu, gamma, pca_d] without cross-validation

C = direct_test(1); nu = direct_test(2); gamma = direct_test(3); pca_d = direct_test(4);


Sig_Ytr = attr_space(unique(Y_train), :);

%% Starting of EXEM training
% SVR kernel
Ker_tr = [(1 : length(unique(Y_train)))', exp(-gamma * pdist2_fast(Sig_Ytr, Sig_Ytr) .^ 2)];
mean_Xtr = compute_class_stat(Y_train, X_train);
regressors = cell(1, pca_d);

for j = 1 : pca_d
    % SVR learning and testing
    regressors{j} = svmtrain(mean_Xtr(:, j), Ker_tr, ['-s 4 -t 4 -c ' num2str(C) ' -n ' num2str(nu) ' -m 10000 -q']);
end
                    
end
