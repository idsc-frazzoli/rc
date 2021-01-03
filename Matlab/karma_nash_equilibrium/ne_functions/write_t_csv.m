% Wrtie transition matrix to csv file
function num_K = write_t_csv(t_down_u_k_up_un_kn, s_up_k, U, K, s_tol, fileprefix)
    num_U = length(U);
    num_K = length(K);

    % Remove tail of distribution where there are too few agents
    while s_up_k(num_K) < s_tol
        num_K = num_K - 1;
    end

    % Header
    header = ["u", "k", "k2", "kn", "kn2", "T"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));

    % Data
    for i_u = 1 : num_U
        u = U(i_u);
        for i_k = 1 : num_K + 1
            k = i_k - 1;
            for i_kn = 1 : num_K + 1
                kn = i_kn - 1;
                line = [u, k, k - 0.5, kn, kn - 0.5, sum(t_down_u_k_up_un_kn(i_u,i_k,:,i_kn))];
                dlmwrite(filename, line, '-append');
            end
        end
    end
end