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

%% Select alpha(s) to process
while true
    k_ave_vec = input('Enter k_ave(s): ');
    
    for i_k_ave = 1 : length(k_ave_vec)
        k_ave = k_ave_vec(i_k_ave);

        % Load workspace for computation for that alpha
        workspace = load(['karma_nash_equilibrium/results/k_ave_', num2str(k_ave, '%02d'), '.mat']);
        ne_param = workspace.ne_param;
        sw_pi_down_u_k_up_m = workspace.sw_pi_down_u_k_up_m;
        sw_d_up_u_k = workspace.sw_d_up_u_k;
        sw_q_down_u_k = workspace.sw_q_down_u_k;
        sw_t_down_u_k_up_un_kn = workspace.sw_t_down_u_k_up_un_kn;

        % Plots
        % SW policy plot
        sw_pi_plot_fg = 1;
        sw_pi_plot_pos = [0, 0, screenwidth, screenheight / 2];
        ne_func.plot_sw_pi(sw_pi_plot_fg, sw_pi_plot_pos, RedColormap, sw_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, k_ave);

        % SW stationary distribution plot
        sw_d_plot_fg = 2;
        sw_d_plot_pos = [0, screenheight / 2, screenwidth, screenheight / 2];
        ne_func.plot_sw_d(sw_d_plot_fg, sw_d_plot_pos, sw_d_up_u_k, ne_param.U, ne_param.K, k_ave);

        % SW expected stage cost plot
        sw_q_plot_fg = 3;
        sw_q_plot_pos = [0, 0, screenwidth / 2, screenheight / 2];
        ne_func.plot_sw_q(sw_q_plot_fg, sw_q_plot_pos, sw_q_down_u_k, ne_param.U, ne_param.K, k_ave);

        % SW state transitions plot
        sw_t_plot_fg = 4;
        sw_t_plot_pos = [0, 0, screenwidth, screenheight];
        ne_func.plot_sw_t(sw_t_plot_fg, sw_t_plot_pos, RedColormap, sw_t_down_u_k_up_un_kn, ne_param.U, ne_param.K, k_ave);
    
        if save_data
            if ~exist('karma_nash_equilibrium/results/policies', 'dir')
                mkdir('karma_nash_equilibrium/results/policies');
            end
            if ~exist('karma_nash_equilibrium/results/stationary_distributions', 'dir')
                mkdir('karma_nash_equilibrium/results/stationary_distributions');
            end
            if ~exist('karma_nash_equilibrium/results/expected_costs', 'dir')
                mkdir('karma_nash_equilibrium/results/expected_costs');
            end
            if ~exist('karma_nash_equilibrium/results/state_transitions', 'dir')
                mkdir('karma_nash_equilibrium/results/state_transitions');
            end

            save(['karma_nash_equilibrium/results/policies/k_ave_', num2str(k_ave, '%02d'), '.mat'], 'sw_pi_down_u_k_up_m');
            save(['karma_nash_equilibrium/results/stationary_distributions/k_ave_', num2str(k_ave, '%02d'), '.mat'], 'sw_d_up_u_k');
            save(['karma_nash_equilibrium/results/expected_costs/k_ave_', num2str(k_ave, '%02d'), '.mat'], 'sw_q_down_u_k');
            save(['karma_nash_equilibrium/results/state_transitions/k_ave_', num2str(k_ave, '%02d'), '.mat'], 'sw_t_down_u_k_up_un_kn');

            saveas(sw_pi_plot_fg, ['karma_nash_equilibrium/results/policies/k_ave_', num2str(k_ave, '%02d'), '.png']);
            saveas(sw_d_plot_fg, ['karma_nash_equilibrium/results/stationary_distributions/k_ave_', num2str(k_ave, '%02d'), '.png']);
            saveas(sw_q_plot_fg, ['karma_nash_equilibrium/results/expected_costs/k_ave_', num2str(k_ave, '%02d'), '.png']);
            saveas(sw_t_plot_fg, ['karma_nash_equilibrium/results/state_transitions/k_ave_', num2str(k_ave, '%02d'), '.png']);
        end
    end
end