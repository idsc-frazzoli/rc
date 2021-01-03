% Wrtie infinite horizon payoff to csv file
function num_K = write_J_csv(v_down_u_k, s_up_k, ne_param, alpha, s_tol, fileprefix)
    num_K = ne_param.num_K;

    % Remove tail of distribution where there are too few agents
    while s_up_k(num_K) < s_tol
        num_K = num_K - 1;
    end

    % Header
    header = ["u", "k", "J"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Header for baseline random payoffs
    filename_rand = [fileprefix, '_rand.csv'];
    fout = fopen(filename_rand, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Compute infinite horizon payoffs for baseline random
    J_down_u_rand = ne_func.random_inf_horizon_payoff(ne_param, alpha);

    % Data
    for i_u = 1 : ne_param.num_U
        for i_k = 1 : num_K
            line = [ne_param.U(i_u), ne_param.K(i_k), -v_down_u_k(i_u,i_k)];
            dlmwrite(filename, line, '-append');

            line_rand = [ne_param.U(i_u), ne_param.K(i_k), J_down_u_rand(i_u)];
            dlmwrite(filename_rand, line_rand, '-append');
        end
    end
end