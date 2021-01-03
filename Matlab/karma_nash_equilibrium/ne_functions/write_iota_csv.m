% Wrtie karma stationary distribution to csv file
function num_K = write_iota_csv(pi_down_u_k_up_m, d_up_u_k, K, iota_tol, fileprefix)
    num_K = length(K);

    % Compute the probability distribution of the messages
    iota_up_m = dot2(reshape(permute(pi_down_u_k_up_m, [3 1 2]), num_K, []), reshape(d_up_u_k, [], 1), 2, 1);

    % Remove tail of policy where there are too few agents
    while iota_up_m(num_K) < iota_tol
        num_K = num_K - 1;
    end

    % Renormalize
    iota_up_m = iota_up_m / sum(iota_up_m);

    % Header
    header = ["b", "b2", "P(b)"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    data = [K(1:num_K+1), K(1:num_K+1) - 0.5, iota_up_m(1:num_K+1)];
    dlmwrite(filename, data, '-append');
end