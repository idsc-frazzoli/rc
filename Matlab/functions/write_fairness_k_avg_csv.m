% Wrtie fariness vs. k_avg to csv file
function write_fairness_k_avg_csv(k_avg, ne_UF, fileprefix)
    % Make sure we have column vectors
    k_avg = reshape(k_avg, [], 1);
    f = -reshape(ne_UF, [], 1);

    % Header
    header = ["k_avg", "f"];
    filename = [fileprefix, '.csv'];
    fout = fopen(filename, 'w');
    for i = 1 : length(header) - 1
        fprintf(fout, '%s,', header(i));
    end
    fprintf(fout, '%s\n', header(end));
    fclose(fout);

    % Data
    data = [k_avg, f];
    dlmwrite(filename, data, '-append');
end