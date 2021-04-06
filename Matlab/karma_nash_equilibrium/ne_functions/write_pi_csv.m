% Write policy to csv file
function n_k = write_pi_csv(pi_down_mu_alpha_u_k_up_m, sigma_up_k, param, ne_param, pi_tol, sigma_tol, fileprefix)
    n_k = ne_param.n_k;
    
    % Remove tail of distribution where there are too few agents
    while sigma_up_k(n_k) < sigma_tol
        n_k = n_k - 1;
    end

    % Remove 'zero' values
    pi_down_mu_alpha_u_k_up_m(pi_down_mu_alpha_u_k_up_m < pi_tol) = 0;
    for i_mu = 1 : param.n_mu
        for i_alpha = 1 : param.n_alpha
            for i_u = 1 : param.n_u
                for i_k = 1 : n_k
                    pi_down_mu_alpha_u_k_up_m(i_mu,i_alpha,i_u,i_k,1:n_k) = pi_down_mu_alpha_u_k_up_m(i_mu,i_alpha,i_u,i_k,1:n_k) / sum(pi_down_mu_alpha_u_k_up_m(i_mu,i_alpha,i_u,i_k,1:n_k));
                end
            end
        end
    end

    % Header
    header = ["mu", "alpha", "u", "k", "k2", "b", "b2", "P(b)"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Header for mean of policy
    header_mean = ["mu", "alpha", "u", "k", "b"];
    filename_mean = [fileprefix, '_mean.csv'];
    fout = fopen(filename_mean, 'w');
    for i = 1 : length(header_mean) - 1
        fprintf(fout, '%s,', header_mean(i));
    end
    fprintf(fout, '%s\n', header_mean(end));
    fclose(fout);

    % Data
    for i_mu = 1 : param.n_mu
        for i_alpha = 1 : param.n_alpha
            alpha = param.Alpha(i_alpha);
            for i_u = 1 : param.n_u
                u = param.U(i_u);
                for i_k = 1 : n_k + 1
                    k = i_k - 1;
                    for i_b = 1 : n_k + 1
                        b = i_b - 1;
                        if i_b <= i_k && i_k <= n_k
                            line = [i_mu, alpha, u, k, k - 0.5, b, b - 0.5, pi_down_mu_alpha_u_k_up_m(i_mu,i_alpha,i_u,i_k,i_b)];
                        else
                            line = [i_mu, alpha, u, k, k - 0.5, b, b - 0.5, 2];
                        end
                        dlmwrite(filename, line, '-append');
                    end

                    if i_k <= n_k
                        line_mean = [i_mu, alpha, u, k, dot(squeeze(pi_down_mu_alpha_u_k_up_m(i_mu,i_alpha,i_u,i_k,:)), ne_param.K)];
                        dlmwrite(filename_mean, line_mean, '-append');
                    end
                end
            end
        end
    end
end