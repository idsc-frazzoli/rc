clear;
close all;
clc;

% Flag to save plots
save_plots = true;

%% Select k_max(s) and k_ave(s) to process
while true
    k_ave_vec = input('Enter k_ave(s): ');
    if k_ave_vec == -1  % Code for processing centralized policies and not karma policies
        % Load workspace for simulation for centralized policies
        load('results/centralized_policies.mat');

        % Plots
        do_plots;

        if save_plots
            if ~exist('results/dynamic/performance_comparisons', 'dir')
                mkdir('results/dynamic/performance_comparisons');
            end
            saveas(performance_comparison_fg, 'results/dynamic/performance_comparisons/centralized_policies.png');
        end
    else
        for i_k_ave = 1 : length(k_ave_vec)
            k_ave = k_ave_vec(i_k_ave);

            % Load workspace for simulation for that (k_max,k_ave)
            load(['results/dynamic/k_ave_', num2str(k_ave, '%02d'), '.mat']);

            % Plots
            do_plots;

            if save_plots
                if ~exist('results/dynamic/performance_comparisons', 'dir')
                    mkdir('results/dynamic/performance_comparisons');
                end
                saveas(performance_comparison_fg, ['results/dynamic/performance_comparisons/k_ave_', num2str(k_ave, '%02d'), '.png']);
            end
        end
    end
end