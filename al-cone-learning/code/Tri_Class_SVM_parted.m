function [SVM] = Tri_Class_SVM_parted(x_pos, x_neg, x_null, C_SVM, SVM_Kernel, kernel_param, xp, xm)
% Using two binary SVMs to discriminate between +,- and 0

y_pos = ones(size(x_pos,1),1);
y_null = -1*ones(size(x_null,1),1);
y_neg = -1*ones(size(x_neg,1),1);

if strcmp(SVM_Kernel{1},'polynomial')
    % + vs 0
    SVM_pos_null = fitcsvm([xp;x_pos; x_null;x_neg],[1;y_pos; y_null;y_neg]', 'KernelFunction', SVM_Kernel{1},'Cost',[0,1;C_SVM,0], 'PolynomialOrder',SVM_Kernel{2}, 'BoxConstraint',kernel_param);

    % - vs 0
    SVM_neg_null = fitcsvm([xm;x_pos;x_null;x_neg],[1;-1*y_pos;y_null;-1*y_neg]','KernelFunction', SVM_Kernel{1},'Cost',[0,1;C_SVM,0], 'PolynomialOrder',SVM_Kernel{2},'BoxConstraint',kernel_param);

else
    % + vs 0
    SVM_pos_null = fitcsvm([xp;x_pos; x_null;x_neg],[1;y_pos; y_null;y_neg]', 'KernelFunction', SVM_Kernel,'Cost',[0,1;C_SVM,0],'BoxConstraint',kernel_param);

    % - vs 0
    SVM_neg_null = fitcsvm([xm;x_pos;x_null;x_neg],[1;-1*y_pos;y_null;-1*y_neg]','KernelFunction', SVM_Kernel,'Cost',[0,1;C_SVM,0],'BoxConstraint',kernel_param);
end

SVM = {SVM_pos_null, SVM_neg_null};
end

