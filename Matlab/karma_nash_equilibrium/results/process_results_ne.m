clear;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
load('karma_nash_equilibrium/RedColormap.mat');

file_str = 'karma_nash_equilibrium/results/ne_U_1_10_p_0.80_0.20_0.80_0.20_m_0_no_sat/k_ave_';

% Flag to save plots
save_plots = true;

%% Select k_ave(s) and alpha(s) to process
while true
    k_ave_vec = input('Enter k_ave(s): ');
    alpha_vec = input('Enter alpha(s): ');
    
    for i_k_ave = 1 : length(k_ave_vec)
        k_ave = k_ave_vec(i_k_ave);
        for i_alpha = 1 : length(alpha_vec)
            alpha = alpha_vec(i_alpha);

            % Load workspace for computation for that alpha
            if alpha > 0.99 && alpha < 1
                file = [file_str, num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.3f'), '.mat'];
            else
                file = [file_str, num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.mat'];
            end
            if ~exist(file, 'file')
                continue;
            end
            load(file, 'ne_param', 'ne_pi_down_u_k_up_m', 'ne_d_up_u_k',...
                'ne_v_down_u_k', 'ne_t_down_u_k_up_un_kn', 'ne_s_up_k',...
                'ne_pi_error_hist');
            if alpha == 1
                load(file, 'ne_J_down_u_k');
            end
            
            % Overrides
%             ne_param.U = 1;
%             ne_param.num_U = 1;
            
            % Plots
            close all;
            % NE policy plot
            ne_pi_plot_fg = 1;
            ne_pi_plot_pos = [0, 0, screenwidth, screenheight / 2];
            ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.K, k_ave, alpha);

            % NE stationary distribution plot
            ne_d_plot_fg = 2;
            ne_d_plot_pos = [0, screenheight / 2, screenwidth, screenheight / 2];
            ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_u_k, ne_param.U, ne_param.K, k_ave, alpha);

            % NE expected utility plot
            ne_v_plot_fg = 3;
            ne_v_plot_pos = [0, 0, screenwidth / 2, screenheight / 2];
            if alpha == 1
                ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_J_down_u_k, ne_param.U, ne_param.K, k_ave, alpha);
            else
                ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_v_down_u_k, ne_param.U, ne_param.K, k_ave, alpha);
            end

            % NE state transitions plot
            ne_t_plot_fg = 4;
            ne_t_plot_pos = [0, 0, screenwidth, screenheight];
            ne_func.plot_ne_t(ne_t_plot_fg, ne_t_plot_pos, RedColormap, ne_t_down_u_k_up_un_kn, ne_param.U, ne_param.K, k_ave, alpha);

            % NE stationary karma distribution plot
            ne_karma_d_plot_fg = 5;
            ne_karma_d_plot_pos = [0, screenheight / 2, screenwidth / 2, screenheight / 2];
            ne_func.plot_ne_karma_d(ne_karma_d_plot_fg, ne_karma_d_plot_pos, ne_s_up_k, ne_param.K, k_ave, alpha);
            
            % NE policy error plot
            ne_pi_error_plot_fg = 6;
            ne_pi_error_plot_pos = [0, 0, screenwidth, screenheight / 2];
            ne_func.plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_pi_error_hist, k_ave, alpha);
            
            if save_plots
                if ~exist('karma_nash_equilibrium/results/policies', 'dir')
                    mkdir('karma_nash_equilibrium/results/policies');
                end
                if ~exist('karma_nash_equilibrium/results/stationary_distributions', 'dir')
                    mkdir('karma_nash_equilibrium/results/stationary_distributions');
                end
                if ~exist('karma_nash_equilibrium/results/expected_utilities', 'dir')
                    mkdir('karma_nash_equilibrium/results/expected_utilities');
                end
                if ~exist('karma_nash_equilibrium/results/state_transitions', 'dir')
                    mkdir('karma_nash_equilibrium/results/state_transitions');
                end
                if ~exist('karma_nash_equilibrium/results/stationary_karma_distributions', 'dir')
                    mkdir('karma_nash_equilibrium/results/stationary_karma_distributions');
                end
                if ~exist('karma_nash_equilibrium/results/ne_policy_errors', 'dir')
                    mkdir('karma_nash_equilibrium/results/ne_policy_errors');
                end
                
                if alpha > 0.99 && alpha < 1
                    saveas(ne_pi_plot_fg, ['karma_nash_equilibrium/results/policies/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.3f'), '.png']);
                    saveas(ne_d_plot_fg, ['karma_nash_equilibrium/results/stationary_distributions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.3f'), '.png']);
                    saveas(ne_v_plot_fg, ['karma_nash_equilibrium/results/expected_utilities/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.3f'), '.png']);
                    saveas(ne_t_plot_fg, ['karma_nash_equilibrium/results/state_transitions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.3f'), '.png']);
                    saveas(ne_karma_d_plot_fg, ['karma_nash_equilibrium/results/stationary_karma_distributions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.3f'), '.png']);
                    saveas(ne_pi_error_plot_fg, ['karma_nash_equilibrium/results/ne_policy_errors/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.3f'), '.png']);
                else
                    saveas(ne_pi_plot_fg, ['karma_nash_equilibrium/results/policies/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                    saveas(ne_d_plot_fg, ['karma_nash_equilibrium/results/stationary_distributions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                    saveas(ne_v_plot_fg, ['karma_nash_equilibrium/results/expected_utilities/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                    saveas(ne_t_plot_fg, ['karma_nash_equilibrium/results/state_transitions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                    saveas(ne_karma_d_plot_fg, ['karma_nash_equilibrium/results/stationary_karma_distributions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                    saveas(ne_pi_error_plot_fg, ['karma_nash_equilibrium/results/ne_policy_errors/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                end
            end
        end
    end
end