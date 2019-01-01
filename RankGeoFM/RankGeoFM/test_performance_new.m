function [pre, rec] = test_performance_new(filename, U_1, L_1, U_2, F_G, top_k)
    inputFile=sprintf('%s/%s',filename,'tensor_size.txt');
    tsize=dlmread(inputFile);
    M = tsize(1); N = tsize(2);
    
    inputFile=sprintf('%s/%s',filename,'train_tensor.txt');
    train_vec=dlmread(inputFile);
    visited_locations = sparse(train_vec(:,1), train_vec(:,2), train_vec(:,3), M, N);
    visited_locations = 2 * (visited_locations>0);
    
    inputFile=sprintf('%s/%s',filename,'tune_tensor.txt');
    test_vec=dlmread(inputFile);
    ground_truth = sparse(test_vec(:,1), test_vec(:,2), test_vec(:,3), M, N);
    ground_truth = ground_truth > 0;
    all_pre = 0;
    all_rec = 0;
    for i = 1:M;
        res = (U_1(i, :) * L_1' + U_2(i, :) * F_G') .* (1 - full(visited_locations(i, :)));
        [~, predicted] = sort(res, 'descend');
        predicted = predicted(1:top_k);
        actual = ground_truth(i, :);
        success = sum(actual(predicted));
        all_pre = all_pre + success / top_k;
        all_rec = all_rec + success / sum(actual);
    end
    pre = all_pre / M;
    rec = all_rec / M;
    fprintf('@%d,pre=%f,recall=%f\n', top_k, pre, rec);
end

