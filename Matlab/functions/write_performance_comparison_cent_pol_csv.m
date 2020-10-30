% Wrtie performance comparison of centralized policies to csv file
function write_performance_comparison_cent_pol_csv(IE_rand, UF_rand, IE_u, UF_u, IE_a, UF_a, IE_u_a, UF_u_a, fileprefix)
    % Convention
    % 1 => baseline random
    % 2 => centralized urgency
    % 3 => centralized cost
    % 4 => centralized urgency then cost
    num_cent_pol = 4;
    cent_pol = (1 : num_cent_pol).';

    % Populate column vectors appropriately
    e = zeros(num_cent_pol, 1);
    f = zeros(num_cent_pol, 1);
    e(1) = -IE_rand(end);
    e(2) = -IE_u(end);
    e(3) = -IE_a(end);
    e(4) = -IE_u_a(end);
    f(1) = -UF_rand(end);
    f(2) = -UF_u(end);
    f(3) = -UF_a(end);
    f(4) = -UF_u_a(end);

    % Make vector out of e_opt
    e_opt = -IE_u(end) * ones(num_cent_pol, 1);

    % PoK
    PoK = e ./ e_opt;
    PoK_opt = e_opt ./ e_opt;

    % Header
    header = ["cent_pol", "e", "PoK", "f", "e_opt", "PoK_opt"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    data = [cent_pol, e, PoK, f, e_opt, PoK_opt];
    dlmwrite(filename, data, '-append');
end