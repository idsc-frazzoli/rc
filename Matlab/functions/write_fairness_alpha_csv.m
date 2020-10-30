% Wrtie fairness vs. alpha to csv file
function write_fairness_alpha_csv(alpha, UF_ne, UF_bid_all, UF_sw, UF_u, UF_rand, UF_a, lim_mem_steps, UF_lim_mem_a, fileprefix)
    % Populate column vectors appropriately
    alpha = reshape(alpha, [], 1);
    num_alpha = length(alpha);
    f = zeros(num_alpha, 1);
    for i_alpha = 1 : num_alpha
        f(i_alpha) = -UF_ne{i_alpha}(end);
    end
    if ~any(alpha == 0)
        alpha = [alpha; 0];
        f = [f; -UF_bid_all(end)];
        num_alpha = length(alpha);
    end            

    % Make vectors out of fairness of benchmark policies
    f_sw = -UF_sw(end) * ones(num_alpha, 1);
    f_u = -UF_u(end) * ones(num_alpha, 1);
    f_rand = -UF_rand(end) * ones(num_alpha, 1);
    f_a = -UF_a(end) * ones(num_alpha, 1);

    % Fairness of limited memory policies
    num_lim_mem_steps = length(lim_mem_steps);
    f_lim_mem = zeros(num_alpha, num_lim_mem_steps);
    for i_lim_mem = 1 : num_lim_mem_steps
        f_lim_mem(:,i_lim_mem) = -UF_lim_mem_a{i_lim_mem}(end) * ones(num_alpha, 1);
    end

    % Header
    header = ["alpha", "f", "f_sw", "f_u", "f_rand", "f_a"];
    for i_lim_mem = 1 : num_lim_mem_steps
        header = [header, strcat("f_a_m_", int2str(lim_mem_steps(i_lim_mem)))];
    end
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    data = [alpha, f, f_sw, f_u, f_rand, f_a, f_lim_mem];
    dlmwrite(filename, data, '-append');
end