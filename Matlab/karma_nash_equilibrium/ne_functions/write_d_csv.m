% Wrtie stationary distribution to csv file
function num_K = write_d_csv(d_up_u_k, U, K, s_tol, fileprefix)
    num_U = length(U);
    num_K = length(K);
    s_up_k = sum(d_up_u_k).';

    % Remove tail of distribution where there are too few agents
    while s_up_k(num_K) < s_tol
        num_K = num_K - 1;
    end
    if num_K == length(K)
        K = [K; K(end)+1];
        d_up_u_k = [d_up_u_k, zeros(num_U, 1)];
    end

    % Header
    header = ["u", "k", "k2", "P(k)"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    for i_u = 1 : num_U
        data = [U(i_u) * ones(num_K + 1, 1), K(1:num_K+1), K(1:num_K+1) - 0.5, d_up_u_k(i_u,1:num_K+1).'];
        dlmwrite(filename, data, '-append');
    end
end