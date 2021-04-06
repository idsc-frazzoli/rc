% Wrtie performance comparison of limited memory policies to csv file
function write_performance_comparison_lim_mem_csv(lim_mem_steps, IE_lim_mem_a, UF_lim_mem_a, IE_u, fileprefix)
    % Populate column vectors appropriately
    lim_mem_steps = reshape(lim_mem_steps, [], 1);
    num_lim_mem_steps = length(lim_mem_steps);
    e = zeros(num_lim_mem_steps, 1);
    f = zeros(num_lim_mem_steps, 1);
    for i_lim_mem = 1 : num_lim_mem_steps
        e(i_lim_mem) = -IE_lim_mem_a{i_lim_mem}(end);
        f(i_lim_mem) = -UF_lim_mem_a{i_lim_mem}(end);
    end

    % Make vector out of e_opt
    e_opt = -IE_u(end) * ones(num_lim_mem_steps, 1);

    % PoK
    PoK = e ./ e_opt;
    PoK_opt = e_opt ./ e_opt;

    % Header
    header = ["m", "e", "PoK", "f", "e_opt", "PoK_opt"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    data = [lim_mem_steps, e, PoK, f, e_opt, PoK_opt];
    dlmwrite(filename, data, '-append');
end