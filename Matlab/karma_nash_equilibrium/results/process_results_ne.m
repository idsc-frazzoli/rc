clear;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
load('karma_nash_equilibrium/RedColormap.mat');

file_str = 'karma_nash_equilibrium/results/ne_U_1_10_phi1_0.50_0.50_0.50_0.50_alpha1_0.30_0.10_alpha2_0.97_0.90_pay_0/k_bar_';

% Flag to save plots
save_plots = true;

%% Select k_bar(s) and alpha(s) to process
while true
    k_bar_vec = input('Enter k_bar(s): ');
    alpha_comp_vec = input('Enter alpha(s): ');
    
    for i_k_bar = 1 : length(k_bar_vec)
        k_bar = k_bar_vec(i_k_bar);
        file_str2 = [file_str, num2str(k_bar, '%02d')];
        for i_alpha_comp = 1 : max([length(alpha_comp_vec), 1])
            if length(alpha_comp_vec) >= 1
                alpha = alpha_comp_vec(i_alpha_comp);
                if alpha > 0.99 && alpha < 1
                    alpha_str = num2str(alpha, '%.3f');
                else
                    alpha_str = num2str(alpha, '%.2f');
                end
                file = [file_str2, '_alpha_', alpha_str, '.mat'];
            else
                file = [file_str2, '.mat'];
            end
            if ~exist(file, 'file')
                continue;
            end
            load(file, 'param', 'ne_param', 'ne_pi_down_mu_alpha_u_k_up_b',...
                'ne_d_up_mu_alpha_u_k', 'br_pi_down_mu_alpha_u_k_up_b',...
                'ne_J_down_mu_alpha_u_k', 'ne_F_down_mu_alpha_u_k_b',...
                'ne_T_down_mu_alpha_u_k_up_un_kn', 'ne_sigma_up_k',...
                'ne_pi_error_hist');
            
            % Plots
            close all;
            % NE policy plot
            ne_pi_plot_fg = 1;
            ne_pi_plot_pos = [0, 0, screenwidth, screenheight];
            plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_mu_alpha_u_k_up_b, param, ne_param, i_alpha_comp);

            % NE stationary distribution plot
            ne_d_plot_fg = 2;
            ne_d_plot_pos = [0, 0, screenwidth, screenheight];
            plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_mu_alpha_u_k, param, ne_param, i_alpha_comp);

            % Best response policy plot
            br_pi_plot_fg = 3;
            br_pi_plot_pos = [0, 0, screenwidth, screenheight];
            plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, br_pi_down_mu_alpha_u_k_up_b, param, ne_param, i_alpha_comp);

            % NE payoffs plot
            ne_J_plot_fg = 4;
            ne_J_plot_pos = [0, 0, screenwidth, screenheight];
            plot_ne_J(ne_J_plot_fg, ne_J_plot_pos, ne_J_down_mu_alpha_u_k, param, ne_param, i_alpha_comp);

            % NE payoffs per bid plot
            ne_F_plot_fg = 5;
            ne_F_plot_pos = [0, 0, screenwidth, screenheight];
            plot_ne_F(ne_F_plot_fg, ne_F_plot_pos, parula, ne_F_down_mu_alpha_u_k_b, param, ne_param, i_alpha_comp);

            % NE state transitions plot
            ne_T_plot_fg = 6;
            ne_T_plot_pos = [0, 0, screenwidth, screenheight];
            plot_ne_T(ne_T_plot_fg, ne_T_plot_pos, RedColormap, ne_T_down_mu_alpha_u_k_up_un_kn, param, ne_param, i_alpha_comp);

            % NE policy error plot
            ne_pi_error_plot_fg = 7;
            ne_pi_error_plot_pos = [0, 0, screenwidth, screenheight];
            plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_pi_error_hist, param, i_alpha_comp);
            
            if save_plots
                if ~exist('karma_nash_equilibrium/results/ne_policies', 'dir')
                    mkdir('karma_nash_equilibrium/results/ne_policies');
                end
                if ~exist('karma_nash_equilibrium/results/ne_distributions', 'dir')
                    mkdir('karma_nash_equilibrium/results/ne_distributions');
                end
                if ~exist('karma_nash_equilibrium/results/ne_payoffs', 'dir')
                    mkdir('karma_nash_equilibrium/results/ne_payoffs');
                end
                if ~exist('karma_nash_equilibrium/results/ne_state_transitions', 'dir')
                    mkdir('karma_nash_equilibrium/results/ne_state_transitions');
                end
                if ~exist('karma_nash_equilibrium/results/ne_policy_errors', 'dir')
                    mkdir('karma_nash_equilibrium/results/ne_policy_errors');
                end
                
                ne_pi_file = ['karma_nash_equilibrium/results/ne_policies/k_bar_', num2str(k_bar, '%02d')];
                ne_d_file = ['karma_nash_equilibrium/results/ne_distributions/k_bar_', num2str(k_bar, '%02d')];
                ne_J_file = ['karma_nash_equilibrium/results/ne_payoffs/k_bar_', num2str(k_bar, '%02d')];
                ne_T_file = ['karma_nash_equilibrium/results/ne_state_transitions/k_bar_', num2str(k_bar, '%02d')];
                ne_pi_error_file = ['karma_nash_equilibrium/results/ne_policy_errors/k_bar_', num2str(k_bar, '%02d')];
                if length(alpha_comp_vec) >= 1
                    ne_pi_file = [ne_pi_file, '_alpha_', alpha_str, '.png'];
                    ne_d_file = [ne_d_file, '_alpha_', alpha_str, '.png'];
                    ne_J_file = [ne_J_file, '_alpha_', alpha_str, '.png'];
                    ne_T_file = [ne_T_file, '_alpha_', alpha_str, '.png'];
                    ne_pi_error_file = [ne_pi_error_file, '_alpha_', alpha_str, '.png'];
                else
                    ne_pi_file = [ne_pi_file, '.png'];
                    ne_d_file = [ne_d_file, '.png'];
                    ne_J_file = [ne_J_file, '.png'];
                    ne_T_file = [ne_T_file, '.png'];
                    ne_pi_error_file = [ne_pi_error_file, '.png'];
                end
                saveas(ne_pi_plot_fg, ne_pi_file);
                saveas(ne_d_plot_fg, ne_d_file);
                saveas(ne_J_plot_fg, ne_J_file);
                saveas(ne_T_plot_fg, ne_T_file);
                saveas(ne_pi_error_plot_fg, ne_pi_error_file);
            end
        end
    end
end