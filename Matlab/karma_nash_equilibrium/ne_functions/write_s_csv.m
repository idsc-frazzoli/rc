% Wrtie stationary karma distribution to csv file
function num_K = write_s_csv(s_up_k, K, s_tol, fileprefix)
    num_K = length(K);

    % Remove tail of distribution where there are too few agents
    while s_up_k(num_K) < s_tol
        num_K = num_K - 1;
    end
    if num_K == length(K)
        K = [K; K(end)+1];
        s_up_k = [s_up_k; 0];
    end

    % Renormalize
    s_up_k = s_up_k / sum(s_up_k);

    % Header
    header = ["k", "k2", "P(k)"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    data = [K(1:num_K+1), K(1:num_K+1) - 0.5, s_up_k(1:num_K+1)];
    dlmwrite(filename, data, '-append');
end