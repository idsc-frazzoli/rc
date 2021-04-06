% Write performance comparison to csv file
function write_performance_comparison_csv(alpha, IE_ne, UF_ne, IE_bid_all, UF_bid_all, IE_sw, UF_sw, IE_u, IE_rand, fileprefix)
    % Populate column vectors appropriately
    alpha = reshape(alpha, [], 1);
    num_alpha = length(alpha);
    e = zeros(num_alpha, 1);
    f = zeros(num_alpha, 1);
    for i_alpha = 1 : num_alpha
        e(i_alpha) = -IE_ne{i_alpha}(end);
        f(i_alpha) = -UF_ne{i_alpha}(end);
    end
    if ~any(alpha == 0)
        alpha = [alpha; 0];
        e = [e; -IE_bid_all(end)];
        f = [f; -UF_bid_all(end)];
        num_alpha = length(alpha);
    end            

    % Make vectors out of e_sw, f_sw, e_opt, e_rand
    e_sw = -IE_sw(end) * ones(num_alpha, 1);
    f_sw = -UF_sw(end) * ones(num_alpha, 1);
    e_opt = -IE_u(end) * ones(num_alpha, 1);
    e_rand = -IE_rand(end) * ones(num_alpha, 1);

    % PoK
    PoK = e ./ e_opt;
    PoK_sw = e_sw ./ e_opt;
    PoK_opt = e_opt ./ e_opt;
    PoK_rand = e_rand ./ e_opt;

    % Header
    header = ["alpha", "e", "PoK", "f", "e_sw", "PoK_sw", "f_sw", "e_opt", "PoK_opt", "e_rand", "PoK_rand"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    data = [alpha, e, PoK, f, e_sw, PoK_sw, f_sw, e_opt, PoK_opt, e_rand, PoK_rand];
    dlmwrite(filename, data, '-append');
end