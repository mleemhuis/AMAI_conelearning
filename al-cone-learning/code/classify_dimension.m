function [SVM] = classify_dimension(regressors, attr_space, attr_space_bin, geom_model, Sig_Ytr, gamma, C_SVM, SVM_Kernel, kernel_param)
    % d regressors (one per dimension of the space)
    % attr_space and attr_space_bin based on the unknown labels which need
    % to be classified
    % Sig_Ytr depicts attr_space of known labels
    
    % Get visual exemplars for unknown and nearby instances

    % number concepts
    n_conc = size(attr_space, 1);
    % number attributs
    n_att = size(attr_space, 2);

    % number of dimensions
    pca_d = length(regressors);

    % classical rbf-kernel for prediction (for the labels of the
    % attr_space, based on known labels Sig_Ytr)
    Ker = [(1 : n_conc)', exp(-gamma * pdist2_fast(attr_space, Sig_Ytr) .^ 2)];

    X_n = zeros(n_conc, pca_d);
    
    for i=1:pca_d
        % prediction of position in feature space
        X_n(:, i) = svmpredict(zeros(n_conc, 1), Ker, regressors{i}, '-q');
    end

    % for each dimension of the model, a classifier is trained
    SVM = cell(1, ceil(n_conc/2));

   
    for i=1:ceil(n_conc/2)
        % First, creation of additional positive and negative instances by
        % adding artificical instances representing partial information
       
        % find index (label) being pos/neg in this dimension
        pos_label = find(geom_model(:,i)==1);
        neg_label = find(geom_model(:,i)==-1);

        tmp_add_pos = [];
        tmp_add_neg=[];
  

        % Several possibilities for getting al-cone representations for
        % partial information. Here, unions of five
        % attribute-representations are considered and added to the
        % representations used for training
        for j=1:n_conc
            for k=j:n_conc
                for l=k:n_conc
                   for m=l:n_conc
                        pos_xor = attr_space_bin(pos_label, :)+attr_space_bin(j,:)+attr_space_bin(k,:)+attr_space_bin(l,:)+attr_space_bin(m,:);
                        neg_xor = attr_space_bin(neg_label, :)+attr_space_bin(j,:)+attr_space_bin(k,:)+attr_space_bin(l,:)+attr_space_bin(m,:);
            
                        pos_xor(pos_xor==5|pos_xor==0) = 0;
                        pos_xor(pos_xor>0)=1;
            
                        neg_xor(neg_xor==5|neg_xor==0) = 0;
                        neg_xor(neg_xor>0)=1;

                        tmp_add_pos = [tmp_add_pos;cast(pos_xor,'logical')];
                        tmp_add_neg = [tmp_add_neg;cast(neg_xor,'logical')];
                   end
                end
            end
        end

        tmp_add_pos = unique(tmp_add_pos,'rows');
        tmp_add_neg = unique(tmp_add_neg,'rows');

        y_add_pos = zeros(size(tmp_add_pos));
        y_add_neg = zeros(size(tmp_add_neg));

        % The new 'partial information' elements are created by changing
        % the existing attributes for the positive elements. Each attribute
        % being positive(negative) originally but tmp_add_pos is
        % negative(positive) at this position is set to 0, meaning the
        % information about this attribute is considered to be partial.
        for ind = 1:size(tmp_add_pos,1)
            y_add_pos(ind,:) = attr_space(pos_label,:);
            y_add_pos(ind,cast(tmp_add_pos(ind,:),'logical'))=0; 
        end
        for ind = 1:size(tmp_add_neg,1)
            y_add_neg(ind,:) = attr_space(neg_label,:);
            y_add_neg(ind,cast(tmp_add_neg(ind,:),'logical'))=0;
        end
            
        n_add_pos = size(y_add_pos, 1);
        n_add_neg = size(y_add_neg, 1);

        % determination of visual exemplars
        Ker_pos = [(1 : n_add_pos)', exp(-gamma * pdist2_fast(y_add_pos, Sig_Ytr) .^ 2)];
        Ker_neg = [(1 : n_add_neg)', exp(-gamma * pdist2_fast(y_add_neg, Sig_Ytr) .^ 2)];

        X_add_pos = zeros(n_add_pos, pca_d);
        X_add_neg = zeros(n_add_neg, pca_d);

        % Calculate regressor for each dimension
        for j=1:pca_d
            % prediction 
            X_add_pos(:, j) = svmpredict(zeros(n_add_pos, 1), Ker_pos, regressors{j}, '-q');
            X_add_neg(:, j) = svmpredict(zeros(n_add_neg, 1), Ker_neg, regressors{j}, '-q');
            
        end

        X_null = X_n(geom_model(:,i)==0, :);

        xp = X_n(pos_label,:);
        xm = X_n(neg_label,:);

        %% Train SVM
        SVM{i} = Tri_Class_SVM_parted(X_add_pos, X_add_neg, X_null, C_SVM, SVM_Kernel, kernel_param, xp, xm);   
    end
end



