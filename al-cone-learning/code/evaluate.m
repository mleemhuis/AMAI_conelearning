function [acc, prec, avg_answer, hits_at_1,avg_answer_1, prec_rec] = evaluate(X,Y, geom_model, SVM)
    plus = 1;
    minus = -1;
    u = -2;
    nullV = 0;
    
    n_el = size(Y,1);
    
    result_vec = zeros(size(X, 1), size(geom_model, 2));

    % prediction for each dimension of the geometric model
    for i=1:size(geom_model,2)
          l1 = predict(SVM{i}{1}, X);
          l2 = predict(SVM{i}{2}, X);
          
          result_vec(l1==1&l2==1,i) = u;
          result_vec(l1==1&l2==-1,i) = plus;
          result_vec(l1==-1&l2==1,i) = minus;
          result_vec(l1==-1&l2==-1,i) = nullV;         
    end
        
    % Get predicted labels as incidence matrix
    y_pred_matrix = zeros(size(result_vec, 1), size(result_vec, 2)*2);
    for i=1:size(result_vec,1)
       for j=1:size(result_vec, 2)
           if abs(result_vec(i,j)) == 1
               [~, ind] = find(transpose(geom_model(:,j)==result_vec(i,j)));
                y_pred_matrix(i, ind) = 1;
           end
           if result_vec(i,j) == u
               [~, ind] = find(transpose(abs(geom_model(:,j))==1));
                y_pred_matrix(i, ind) = 1;
           end
       end
       
    end
    
    % accuracy
    y_pred = zeros(n_el,1);
    for i=1:n_el
       if y_pred_matrix(i, Y(i)) == 1
           y_pred(i) = Y(i);
       end
    end
    acc= evaluate_easy(y_pred, Y);

    % 2. avg number of answers
    avg_answer = sum(sum(y_pred_matrix))/n_el;

    % 3. hits@1
    y_pred_one = y_pred_matrix;
    y_pred_one(sum(y_pred_one,2)>1,:)=0;

    hits_at_1 = 0;
    for i=1:n_el
        if y_pred_one(i,Y(i))==1
            hits_at_1 = hits_at_1 +1;
        end
    end
    hits_at_1 = hits_at_1/n_el;

    % avg amount of elements having only one result
    avg_answer_1 = sum(sum(y_pred_one))/n_el;

    % calculation of general precision and recall per #of returned labels
    prec_rec = zeros(size(result_vec,2)*2, 2);
    prec = 0;
    n_el_prec = zeros(size(result_vec,2)*2,1);
    for i=1:n_el
        if sum(y_pred_matrix(i,:),2)>0
            n_el_prec(sum(y_pred_matrix(i,:),2):end) = n_el_prec(sum(y_pred_matrix(i,:),2):end)+1;
        end
        if y_pred_matrix(i,Y(i))==1
          prec = prec + 1/sum(y_pred_matrix(i,:),2);
          prec_rec(sum(y_pred_matrix(i,:),2):end, 1) = prec_rec(sum(y_pred_matrix(i,:),2):end, 1) + 1/sum(y_pred_matrix(i,:),2);
          prec_rec(sum(y_pred_matrix(i,:),2):end, 2) = prec_rec(sum(y_pred_matrix(i,:),2):end, 2) + 1;
        end
    end

    % Precision and Recall 
    prec_rec(:,2) = prec_rec(:,2)/n_el;
    prec_rec(:,1) = prec_rec(:,1)./n_el_prec;
    prec = prec/n_el;
  end