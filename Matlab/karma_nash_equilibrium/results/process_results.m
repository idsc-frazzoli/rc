clear;
close all;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
load('karma_nash_equilibrium/RedColormap.mat');

% Flag to save data
save_data = false;

%% Select alpha(s) to process
while true
    alpha_vec = input('Enter alpha(s): ');
    
    for i_alpha = 1 : length(alpha_vec)
    alpha = alpha_vec(i_alpha);
        
    % Load workspace for computation for that alpha
    workspace = load(['karma_nash_equilibrium/results/alpha_', num2str(alpha, '%.2f'), '.mat']);
    ne_param = workspace.ne_param;
    pi_down_u_k_up_m = workspace.pi_down_u_k_up_m;
    D_up_u_k = workspace.D_up_u_k;
    V_down_u_k = workspace.V_down_u_k;
    V_down_u_k_m = workspace.V_down_u_k_m;
    T_down_u_k_up_un_kn = workspace.T_down_u_k_up_un_kn;
    ne_policy_error_hist = workspace.ne_policy_error_hist;
    
    % Plots
    % NE policy plot
    ne_pi_plot_fg = 1;
    ne_pi_plot_pos = [0, 0, screenwidth, screenheight / 2];
    ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, pi_down_u_k_up_m, ne_param.U, ne_param.K, alpha);

    % NE stationary distribution plot
    ne_D_plot_fg = 2;
    ne_D_plot_pos = [0, screenheight / 2, screenwidth, screenheight / 2];
    ne_func.plot_ne_D(ne_D_plot_fg, ne_D_plot_pos, D_up_u_k, ne_param.U, ne_param.K, alpha);
    
    % NE expected utility plot
    ne_V_plot_fg = 3;
    ne_V_plot_pos = [0, 0, screenwidth / 2, screenheight / 2];
    ne_func.plot_ne_V(ne_V_plot_fg, ne_V_plot_pos, V_down_u_k, ne_param.U, ne_param.K, alpha);
    
    % NE expected utiliy per message plot
    ne_V_m_plot_fg = 4;
    ne_V_m_plot_pos = [0, screenheight / 2, screenwidth, screenheight / 2];
    ne_func.plot_ne_V_m(ne_V_m_plot_fg, ne_V_m_plot_pos, parula, V_down_u_k_m, ne_param.U, ne_param.K, alpha);
    
    % NE state transitions plot
    ne_T_plot_fg = 5;
    ne_T_plot_pos = [0, 0, screenwidth, screenheight];
    ne_func.plot_ne_T(ne_T_plot_fg, ne_T_plot_pos, RedColormap, T_down_u_k_up_un_kn, ne_param.U, ne_param.K, alpha);
    
    % NE policy error plot
    ne_pi_error_plot_fg = 6;
    ne_pi_error_plot_pos = [0, 0, screenwidth, screenheight / 2];
    ne_func.plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_policy_error_hist, alpha);
    
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

            save(['karma_nash_equilibrium/results/policies/alpha_', num2str(alpha, '%.2f'), '.mat'], 'pi_down_u_k_up_m');
            save(['karma_nash_equilibrium/results/stationary_distributions/alpha_', num2str(alpha, '%.2f'), '.mat'], 'D_up_u_k');
            save(['karma_nash_equilibrium/results/expected_utilities/alpha_', num2str(alpha, '%.2f'), '.mat'], 'V_down_u_k');
            save(['karma_nash_equilibrium/results/expected_utilities_per_message/alpha_', num2str(alpha, '%.2f'), '.mat'], 'V_down_u_k_m');
            save(['karma_nash_equilibrium/results/state_transitions/alpha_', num2str(alpha, '%.2f'), '.mat'], 'T_down_u_k_up_un_kn');
            save(['karma_nash_equilibrium/results/ne_policy_errors/alpha_', num2str(alpha, '%.2f'), '.mat'], 'ne_policy_error_hist');

            saveas(ne_pi_plot_fg, ['karma_nash_equilibrium/results/policies/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_D_plot_fg, ['karma_nash_equilibrium/results/stationary_distributions/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_V_plot_fg, ['karma_nash_equilibrium/results/expected_utilities/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_V_m_plot_fg, ['karma_nash_equilibrium/results/expected_utilities_per_message/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_T_plot_fg, ['karma_nash_equilibrium/results/state_transitions/alpha_', num2str(alpha, '%.2f'), '.png']);
            saveas(ne_pi_error_plot_fg, ['karma_nash_equilibrium/results/ne_policy_errors/alpha_', num2str(alpha, '%.2f'), '.png']);
        end
    end
end