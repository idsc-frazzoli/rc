clear;
close all;
clc;

%% String to replace
old_string = 'ne_pi_down_u_k_up_b';
new_string = 'ne_pi_down_mu_alpha_u_k_up_b';
extend_types = true;

%% Do the renaming
fileList = dir('**/*.mat');
for i_file = 1 : length(fileList)
    file = fullfile(fileList(i_file).folder, fileList(i_file).name);
    fprintf('Processing file %s\n', file);
    old_mat = load(file);
    old_names = fieldnames(old_mat);
    new_names = strrep(old_names, old_string, new_string); 
    for i_names = 1:length(old_names)
        if extend_types && old_names{i_names} == old_string
            old_mat.(old_names{i_names}) = reshape(old_mat.(old_names{i_names}), [1, 1, size(old_mat.(old_names{i_names}))]);
        end
        new_mat.(new_names{i_names}) = old_mat.(old_names{i_names});
    end
    save(file, '-struct', 'new_mat');
end

fprintf('\nDONE\n');