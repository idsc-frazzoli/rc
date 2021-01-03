% Write policy to csv file
function num_K = write_pi_csv(pi_down_u_k_up_m, s_up_k, U, K, pi_tol, s_tol, fileprefix)
    num_U = length(U);
    num_K = length(K);

    % Remove tail of distribution where there are too few agents
    while s_up_k(num_K) < s_tol
        num_K = num_K - 1;
    end

    % Remove 'zero' values
    pi_down_u_k_up_m(pi_down_u_k_up_m < pi_tol) = 0;
    for i_u = 1 : num_U
        for i_k = 1 : num_K
            pi_down_u_k_up_m(i_u,i_k,1:num_K) = pi_down_u_k_up_m(i_u,i_k,1:num_K) / sum(pi_down_u_k_up_m(i_u,i_k,1:num_K));
        end
    end

    % Header
    header = ["u", "k", "k2", "b", "b2", "P(b)"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Header for mean of policy
    header_mean = ["u", "k", "b"];
    filename_mean = [fileprefix, '_mean.csv'];
    fout = fopen(filename_mean, 'w');
    for i = 1 : length(header_mean) - 1
        fprintf(fout, '%s,', header_mean(i));
    end
    fprintf(fout, '%s\n', header_mean(end));
    fclose(fout);

    % Data
    for i_u = 1 : num_U
        u = U(i_u);
        for i_k = 1 : num_K + 1
            k = i_k - 1;
            for i_b = 1 : num_K + 1
                b = i_b - 1;
                if i_b <= i_k && i_k <= num_K
                    line = [u, k, k - 0.5, b, b - 0.5, pi_down_u_k_up_m(i_u,i_k,i_b)];
                else
                    line = [u, k, k - 0.5, b, b - 0.5, 2];
                end
                dlmwrite(filename, line, '-append');
            end

            if i_k <= num_K
                line_mean = [U(i_u), K(i_k), dot(squeeze(pi_down_u_k_up_m(i_u,i_k,:)), K)];
                dlmwrite(filename_mean, line_mean, '-append');
            end
        end
    end
end