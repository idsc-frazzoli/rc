clear;
close all;
clc;

%% Some parameters
% Test case
file_str = 'results/N_200_T_100000_U_1_2_p_0.50_m_0_no_sat/';

% Plot heuristic policies
karma_heuristic_policies = false;

% Entropy computations
compute_entropy = false;

% Flag to save plots
save_plots = true;

%% Select k_max(s) and k_ave(s) to process
while true
    k_ave_vec = input('Enter k_ave(s): ');
    if k_ave_vec == -1  % Code for processing centralized policies and not karma policies
        % Load workspace for simulation for centralized policies
        file = [file_str, 'centralized_policies.mat'];
        load(file, 'param', 'control', 'IE_*', 'UF_*');

        % Plots
        do_plots;

        if save_plots
            if ~exist('results/performance_comparisons', 'dir')
                mkdir('results/performance_comparisons');
            end
            saveas(performance_comparison_fg, 'results/performance_comparisons/centralized_policies.png');
        end
    else
        for i_k_ave = 1 : length(k_ave_vec)
            k_ave = k_ave_vec(i_k_ave);

            % Load workspace for simulation for that (k_max,k_ave)
            file = [file_str, 'k_ave_', num2str(k_ave, '%02d'), '.mat'];
%             file = [file_str, 'k_ave_', num2str(k_ave, '%02d'), '_alpha_high.mat'];
            load(file, 'param', 'control', 'IE_*', 'UF_*', 'K_ne');
            
            file = [file_str, 'k_ave_', num2str(k_ave, '%02d'), '_alpha_1.00.mat'];
            if exist(file, 'file')
                alpha_1 = load(file, 'param', 'control', 'IE_*', 'UF_*', 'K_ne');
                IE_ne_temp = IE_ne;
                UF_ne_temp = UF_ne;
                K_ne_temp = K_ne;
                IE_ne = cell(param.num_alpha + 1, 1);
                UF_ne = cell(param.num_alpha + 1, 1);
                K_ne = cell(param.num_alpha + 1, 1);
                IE_ne{1} = alpha_1.IE_ne{1};
                UF_ne{1} = alpha_1.UF_ne{1};
                K_ne{1} = alpha_1.K_ne{1};
                for i_alpha = 1 : param.num_alpha
                    IE_ne{i_alpha+1} = IE_ne_temp{i_alpha};
                    UF_ne{i_alpha+1} = UF_ne_temp{i_alpha};
                    K_ne{i_alpha+1} = K_ne_temp{i_alpha};
                    IE_ne_temp{i_alpha} = [];
                    UF_ne_temp{i_alpha} = [];
                    K_ne_temp{i_alpha} = [];
                end
                clear IE_ne_temp UF_ne_temp K_ne_Temp
                param.alpha = [1, param.alpha];
                param.num_alpha = length(param.alpha);
                
%                 file = [file_str, 'k_ave_', num2str(k_ave, '%02d'), '_alpha_high.mat'];
%                 alpha_high = load(file, 'param', 'control', 'IE_*', 'UF_*', 'K_ne');
%                 IE_ne = cell(param.num_alpha + alpha_high.param.num_alpha + 1, 1);
%                 UF_ne = cell(param.num_alpha + alpha_high.param.num_alpha + 1, 1);
%                 K_ne = cell(param.num_alpha + alpha_high.param.num_alpha + 1, 1);
%                 IE_ne{1} = alpha_1.IE_ne{1};
%                 UF_ne{1} = alpha_1.UF_ne{1};
%                 K_ne{1} = alpha_1.K_ne{1};
%                 for i_alpha = 1 : alpha_high.param.num_alpha
%                     IE_ne{i_alpha+1} = alpha_high.IE_ne{i_alpha};
%                     UF_ne{i_alpha+1} = alpha_high.UF_ne{i_alpha};
%                     K_ne{i_alpha+1} = alpha_high.K_ne{i_alpha};
%                 end
%                 for i_alpha = 1 : param.num_alpha
%                     IE_ne{i_alpha+alpha_high.param.num_alpha+1} = IE_ne_temp{i_alpha};
%                     UF_ne{i_alpha+alpha_high.param.num_alpha+1} = UF_ne_temp{i_alpha};
%                     K_ne{i_alpha+alpha_high.param.num_alpha+1} = K_ne_temp{i_alpha};
%                     IE_ne_temp{i_alpha} = [];
%                     UF_ne_temp{i_alpha} = [];
%                     K_ne_temp{i_alpha} = [];
%                 end
%                 clear IE_ne_temp UF_ne_temp K_ne_Temp
%                 param.alpha = [1, alpha_high.param.alpha, param.alpha];
%                 param.num_alpha = length(param.alpha);
            end
            
            % Overrides the alphas of interest (can only do from the
            % 'bottom')
            param.alpha = [1.00 : -0.01 : 0.96, 0.95 : -0.05 : 0.20];
%             param.alpha = [1.00 : -0.001 : 0.99, 0.98 : -0.01 : 0.95];
            param.num_alpha = length(param.alpha);
            
            % Overrides whether to plot karma heuristic policies or not
            control.karma_heuristic_policies = karma_heuristic_policies;
            
            % Overrides whether to diplay entropy or not
            param.compute_entropy = compute_entropy;
            
            % Plots
            do_plots;

            if save_plots
                if ~exist('results/performance_comparisons', 'dir')
                    mkdir('results/performance_comparisons');
                end
                saveas(performance_comparison_fg, ['results/performance_comparisons/k_ave_', num2str(k_ave, '%02d'), '.png']);
            end
        end
    end
end