clear;
close all;
clc;

%% String to replace
old_string = 'up_m';
new_string = 'up_b';

%% Do the renaming
fileList = dir('**/*.mat');
for i_file = 1 : length(fileList)
    file = fullfile(fileList(i_file).folder, fileList(i_file).name);
    fprintf('Processing file %s\n', file);
    old_mat = load(file);
    old_names = fieldnames(old_mat);
    new_names = strrep(old_names, old_string, new_string); 
    for i_names = 1:length(old_names)
        new_mat.(new_names{i_names}) = old_mat.(old_names{i_names});
    end
    save(file, '-struct', 'new_mat');
end

fprintf('\nDONE\n');