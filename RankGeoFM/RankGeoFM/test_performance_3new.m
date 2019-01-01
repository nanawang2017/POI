function [] = test_performance_3new(filename, top_k)
    inputFile=sprintf('%s/%s',filename,'tensor_size.txt');
    tsize=dlmread(inputFile);
    M = tsize(1); N = tsize(2);
    inputFile=sprintf('%s/%s',filename,'train_tensor.txt');
    train_vec=dlmread(inputFile);
    visited_locations = sparse(train_vec(:,1), train_vec(:,2), train_vec(:,3), M, N);
    visited_locations = 2 * (visited_locations>0);

    inputFile=sprintf('%s_POI_Rec_Iterations/%s',filename,'decomp-WARP.mat');
    load(inputFile);
    [~,K]=size(U_1);
    [~,k_1]=size(Index_A);
    if(k_1>0)
        F_G=zeros(tsize(2),K);
        for i=1:tsize(2)
            for j=1:k_1
                F_G(i,:)=F_G(i,:)+A(i,j)*L_1(Index_A(i,j),:);
            end
        end
    end
    
    fout=fopen('./result/sigir15li_100.dat', 'w');
    for i = 1:M;
        if mod(i, 100) == 0
            disp(i);
        end
        res = (U_1(i, :) * L_1' + U_2(i, :) * F_G') .* (1 - full(visited_locations(i, :)));
        [score, predicted] = sort(res, 'descend');
        predicted = predicted(1:top_k);
        fprintf(fout, '%d', i);
        fprintf(fout, ' %d', predicted(1));
        fprintf(fout, ',%d', predicted(2:top_k));
        fprintf(fout, '\n');
    end
    fclose(fout);
end

