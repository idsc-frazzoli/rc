% Wrtie stationary distribution to csv file
function n_k = write_d_csv(d_up_mu_alpha_u_k, sigma_up_k, param, ne_param, sigma_tol, fileprefix)
    n_k = ne_param.n_k;
    K = ne_param.K;

    % Remove tail of distribution where there are too few agents
    while sigma_up_k(n_k) < sigma_tol
        n_k = n_k - 1;
    end
    if n_k == length(K)
        K(end+1) = K(end)+1;
        d_up_mu_alpha_u_k(:,:,:,end+1) = 0;
    end

    % Header
    header = ["mu", "alpha", "u", "k", "k2", "P(k)"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    for i_mu = 1 : param.n_mu
        MU = i_mu * ones(n_k + 1, 1);
        for i_alpha = 1 : param.n_alpha
            ALPHA = param.Alpha(i_alpha) * ones(n_k + 1, 1);
            for i_u = 1 : param.n_u
                data = [MU, ALPHA, param.U(i_u) * ones(n_k + 1, 1), K(1:n_k+1), K(1:n_k+1) - 0.5, squeeze(d_up_mu_alpha_u_k(i_mu,i_alpha,i_u,1:n_k+1))];
                dlmwrite(filename, data, '-append');
            end
        end
    end
end