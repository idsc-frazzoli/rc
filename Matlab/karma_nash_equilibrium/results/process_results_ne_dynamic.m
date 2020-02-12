clear;
close all;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
load('karma_nash_equilibrium/RedColormap.mat');

% Flag to save data
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
            workspace = load(['karma_nash_equilibrium/results/k_ave_' num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.mat']);
            ne_param = workspace.ne_param;
            ne_s_up_k = workspace.ne_s_up_k;
            while ne_s_up_k(end) < 1e-3
                ne_s_up_k(end) = [];
            end
            num_K = length(ne_s_up_k);
            K = 0 : num_K - 1;
            ne_pi_down_u_k_up_m = workspace.ne_pi_down_u_k_up_m;
            ne_pi_down_u_k_up_m(:,num_K+1:end,:) = [];
            ne_pi_down_u_k_up_m(:,:,num_K+1:end) = [];
            ne_d_up_u_k = workspace.ne_d_up_u_k;
            ne_d_up_u_k(:,num_K+1:end) = [];
            ne_v_down_u_k = workspace.ne_v_down_u_k;
            ne_v_down_u_k(:,num_K+1:end) = [];
            ne_t_down_u_k_up_un_kn = workspace.ne_t_down_u_k_up_un_kn;

            % Plots
            % NE policy plot
            ne_pi_plot_fg = 1;
            ne_pi_plot_pos = [0, 0, screenwidth, screenheight / 2];
            ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, K, K, k_ave, alpha);

            % NE stationary distribution plot
            ne_d_plot_fg = 2;
            ne_d_plot_pos = [0, screenheight / 2, screenwidth, screenheight / 2];
            ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_u_k, ne_param.U, K, k_ave, alpha);

            % NE expected utility plot
            ne_v_plot_fg = 3;
            ne_v_plot_pos = [0, 0, screenwidth / 2, screenheight / 2];
            ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_v_down_u_k, ne_param.U, K, k_ave, alpha);

            % NE state transitions plot
            ne_t_plot_fg = 4;
            ne_t_plot_pos = [0, 0, screenwidth, screenheight];
            ne_func.plot_ne_t(ne_t_plot_fg, ne_t_plot_pos, RedColormap, ne_t_down_u_k_up_un_kn, ne_param.U, workspace.K, k_ave, alpha);

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
                saveas(ne_pi_plot_fg, ['karma_nash_equilibrium/results/policies/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                saveas(ne_d_plot_fg, ['karma_nash_equilibrium/results/stationary_distributions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                saveas(ne_v_plot_fg, ['karma_nash_equilibrium/results/expected_utilities/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                saveas(ne_t_plot_fg, ['karma_nash_equilibrium/results/state_transitions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
            end
            
            close all;
        end
    end
end