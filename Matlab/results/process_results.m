clear;
close all;
clc;

% Flag to save plots
save_plots = true;

%% Select k_max(s) and k_ave(s) to process
while true
    k_max_vec = input('Enter k_max(s): ');
    k_ave_vec = input('Enter k_ave(s): ');
    
    for i_k_max = 1 : length(k_max_vec)
        k_max = k_max_vec(i_k_max);
        for i_k_ave = 1 : length(k_ave_vec)
            k_ave = k_ave_vec(i_k_ave);
            
            % Load workspace for computation for that alpha
            load(['results/k_max_', num2str(k_max, '%02d'), '_k_ave_', num2str(k_ave, '%02d'), '.mat']);

            % Plots
            do_plots;

            if save_plots
                if ~exist('results/performance_comparisons', 'dir')
                    mkdir('results/performance_comparisons');
                end
                saveas(performance_comparison_fg, ['results/performance_comparisons/k_max_', num2str(k_max, '%02d'), '_k_ave_', num2str(k_ave, '%02d'), '.png']);
            end
        end
    end
end