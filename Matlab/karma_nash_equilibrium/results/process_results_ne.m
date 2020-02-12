clear;
close all;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
load('karma_nash_equilibrium/RedColormap.mat');

% Flag to save data
save_data = true;

% Old vs. new naming convention
new_names = true;

%% Select alpha(s) to process
while true
    alpha_vec = input('Enter alpha(s): ');
    
    for i_alpha = 1 : length(alpha_vec)
        alpha = alpha_vec(i_alpha);

        % Load workspace for computation for that alpha
        workspace = load(['karma_nash_equilibrium/results/alpha_', num2str(alpha, '%.2f'), '.mat']);
        ne_param = workspace.ne_param;
        if new_names
            ne_pi_down_u_k_up_m = workspace.ne_pi_down_u_k_up_m;
            ne_d_up_u_k = workspace.ne_d_up_u_k;
            ne_v_down_u_k = workspace.ne_v_down_u_k;
            ne_rho_down_u_k_m = workspace.ne_rho_down_u_k_m;
            ne_t_down_u_k_up_un_kn = workspace.ne_t_down_u_k_up_un_kn;
            ne_pi_error_hist = workspace.ne_pi_error_hist;
        else
            ne_pi_down_u_k_up_m = workspace.pi_down_u_k_up_m;
            ne_d_up_u_k = workspace.D_up_u_k;
            ne_v_down_u_k = workspace.V_down_u_k;
            ne_rho_down_u_k_m = workspace.V_down_u_k_m;
            ne_t_down_u_k_up_un_kn = workspace.T_down_u_k_up_un_kn;
            ne_pi_error_hist = workspace.ne_policy_error_hist;
        end

        % Plots
        % NE policy plot
        ne_pi_plot_fg = 1;
        ne_pi_plot_pos = [0, 0, screenwidth, screenheight / 2];
        ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);

        % NE stationary distribution plot
        ne_d_plot_fg = 2;
        ne_d_plot_pos = [0, screenheight / 2, screenwidth, screenheight / 2];
        ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);

        % NE expected utility plot
        ne_v_plot_fg = 3;
        ne_v_plot_pos = [0, 0, screenwidth / 2, screenheight / 2];
        ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_v_down_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);

        % NE expected utiliy per message plot
        ne_rho_plot_fg = 4;
        ne_rho_plot_pos = [0, screenheight / 2, screenwidth, screenheight / 2];
        ne_func.plot_ne_rho(ne_rho_plot_fg, ne_rho_plot_pos, parula, ne_rho_down_u_k_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);

        % NE state transitions plot
        ne_t_plot_fg = 5;
        ne_t_plot_pos = [0, 0, screenwidth, screenheight];
        ne_func.plot_ne_t(ne_t_plot_fg, ne_t_plot_pos, RedColormap, ne_t_down_u_k_up_un_kn, ne_param.U, ne_param.K, ne_param.k_ave, alpha);

        % NE policy error plot
        ne_pi_error_plot_fg = 6;
        ne_pi_error_plot_pos = [0, 0, screenwidth, screenheight / 2];
        ne_func.plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_pi_error_hist, ne_param.k_ave, alpha);
    
        if save_data
            if ~exist('karma_nash_equilibrium/results/policies', 'dir')
                mkdir('karma_nash_equilibrium/results/policies');
            end
            if ~exist('karma_nash_equilibrium/results/stationary_distributions', 'dir')
                mkdir('karma_nash_equilibrium/results/stationary_distributions');
            end
            if ~exist('karma_nash_equilibrium/results/expected_utilities', 'dir')
                mkdir('karma_nash_equilibrium/results/expected_utilities');
            end
            if ~exist('karma_nash_equilibrium/results/expected_utilities_per_message', 'dir')
                mkdir('karma_nash_equilibrium/results/expected_utilities_per_message');
            end
            if ~exist('karma_nash_equilibrium/results/state_transitions', 'dir')
                mkdir('karma_nash_equilibrium/results/state_transitions');
            end
            if ~exist('karma_nash_equilibrium/results/ne_policy_errors', 'dir')
                mkdir('karma_nash_equilibrium/results/ne_policy_errors');
            end

            save(['karma_nash_equilibrium/results/policies/alpha_', num2str(alpha, '%.2f'), '.mat'], 'ne_pi_down_u_k_up_m');
            save(['karma_nash_equilibrium/results/stationary_distributions/alpha_', num2str(alpha, '%.2f'), '.mat'], 'ne_d_up_u_k');
            save(['karma_nash_equilibrium/results/expected_utilities/alpha_', num2str(alpha, '%.2f'), '.mat'], 'ne_v_down_u_k');
            save(['karma_nash_equilibrium/results/expected_utilities_per_message/alpha_', num2str(alpha, '%.2f'), '.mat'], 'ne_rho_down_u_k_m');
            save(['karma_nash_equilibrium/results/state_transitions/alpha_', num2str(alpha, '%.2f'), '.mat'], 'ne_t_down_u_k_up_un_kn');
            save(['karma_nash_equilibrium/results/ne_policy_errors/alpha_', num2str(alpha, '%.2f'), '.mat'], 'ne_pi_error_hist');

            saveas(ne_pi_plot_fg, ['karma_nash_equilibrium/results/policies/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_d_plot_fg, ['karma_nash_equilibrium/results/stationary_distributions/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_v_plot_fg, ['karma_nash_equilibrium/results/expected_utilities/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_rho_plot_fg, ['karma_nash_equilibrium/results/expected_utilities_per_message/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_t_plot_fg, ['karma_nash_equilibrium/results/state_transitions/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_pi_error_plot_fg, ['karma_nash_equilibrium/results/ne_policy_errors/alpha_', num2str(alpha, '%.2f'), '.png']);
        end
    end
end