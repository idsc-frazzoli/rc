% Wrtie price of karma vs. alpha to csv file
function write_PoK_alpha_csv(alpha_vec, e, e_opt, e_rand, fileprefix)
    % Make sure we have column vectors
    alpha_vec = reshape(alpha_vec, [], 1);
    e = reshape(e, [], 1);

    % Add efficiency for alpha = 0, which is baseline random
    % efficiency
    e(alpha_vec == 0) = [];
    alpha_vec(alpha_vec == 0) = [];
    alpha_vec = [0; alpha_vec];
    e = [e_rand; e];

    % Make vectors out of e_opt and e_rand for plotting
    num_Alpha = length(alpha_vec);
    e_opt = e_opt * ones(num_Alpha, 1);
    e_rand = e_rand * ones(num_Alpha, 1);

    % PoK
    PoK = e ./ e_opt;
    PoK_opt = e_opt ./ e_opt;
    PoK_rand = e_rand ./ e_opt;

    % Header
    header = ["alpha", "e", "PoK", "e_opt", "PoK_opt", "e_rand", "PoK_rand"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    data = [alpha_vec, e, PoK, e_opt, PoK_opt, e_rand, PoK_rand];
    dlmwrite(filename, data, '-append');
end