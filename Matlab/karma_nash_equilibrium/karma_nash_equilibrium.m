clear;
close all;
clc;

%% Add functions folder to path
addpath('functions');
addpath('karma_nash_equilibrium/ne_functions');

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
default_width = screenwidth / 2;
default_height = screenheight / 2;
load('karma_nash_equilibrium/RedColormap.mat');

%% Parameters
% Setting parameters
param = load_parameters();
% NE computation parameters
ne_param = load_ne_parameters(param);

%% Iterative algorithm to find Karma Nash equilibrium %%
%% Step 0: Game tensors
[zeta_down_u_b_bj, kappa_down_k_b_up_kn_down_bj] = get_game_tensors(param, ne_param);

% Initial policy
pi_down_mu_alpha_u_k_up_b_init = get_pi_init(param, ne_param);

% Plot
if ne_param.plot
    ne_pi_plot_fg = 1;
    ne_pi_plot_pos = [0, 0, screenwidth, screenheight];
    plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, pi_down_mu_alpha_u_k_up_b_init, param, ne_param, 1);
end

% Initial population distribution
[prob_down_mu_up_u_init, prob_down_mu_alpha_up_k_init, d_up_mu_alpha_u_k_init] = get_d_init(param.k_bar, param, ne_param);

% Plot
if ne_param.plot
    ne_d_plot_fg = 2;
    ne_d_plot_pos = [0, 0, screenwidth, screenheight];
    plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, d_up_mu_alpha_u_k_init, param, ne_param, 1);
end

%% Loop over alphas
i_alpha_comp = 1;
while i_alpha_comp <= param.n_alpha_comp
    if param.n_alpha_comp > 1
        fprintf('%%%%ALPHA = %.3f%%%%\n\n', param.Alpha(i_alpha_comp));
    end

    %% Step 1: NE policy & distribution guess
    % 1.1
    ne_pi_down_mu_alpha_u_k_up_b = pi_down_mu_alpha_u_k_up_b_init;
    ne_d_up_mu_alpha_u_k = d_up_mu_alpha_u_k_init;
    if ne_param.store_hist
        ne_pi_hist = zeros(ne_param.ne_pi_max_iter + 1, param.n_mu * param.n_alpha * param.n_u * ne_param.n_k * ne_param.n_k);
        ne_pi_hist(1,:) = reshape(ne_pi_down_mu_alpha_u_k_up_b, 1, []);
        ne_d_hist = zeros(ne_param.ne_pi_max_iter + 1, param.n_mu * param.n_alpha * param.n_u * ne_param.n_k);
        ne_d_hist(1,:) = reshape(ne_d_up_mu_alpha_u_k, 1, []);
    end
    
    % Plot
    if ne_param.plot
        plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_mu_alpha_u_k_up_b, param, ne_param, i_alpha_comp);
        plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_mu_alpha_u_k, param, ne_param, i_alpha_comp);
    end
    
    num_ne_pi_iter = 1;
    ne_pi_error = inf;
    ne_pi_error_hist = zeros(ne_param.ne_pi_max_iter, 1);
    ne_d_error = inf;
    ne_d_error_hist = zeros(ne_param.ne_pi_max_iter, 1);
    ne_J_down_mu_alpha_u_k = zeros(param.n_mu, param.n_alpha, param.n_u, ne_param.n_k);
    ne_J_down_mu_alpha_u_k_next = zeros(param.n_mu, param.n_alpha, param.n_u, ne_param.n_k);
    k_max_saturated = false;
    while (ne_pi_error > ne_param.ne_pi_tol || ne_d_error > ne_param.ne_d_tol) && num_ne_pi_iter <= ne_param.ne_pi_max_iter
        %% Step 3
        % 3.1
        nu_up_bj = dot2(reshape(permute(ne_pi_down_mu_alpha_u_k_up_b, [5 1 2 3 4]), ne_param.n_k, []), reshape(ne_d_up_mu_alpha_u_k, [], 1), 2, 1);
        cost_down_u_b = dot2(zeta_down_u_b_bj, nu_up_bj, 3, 1);
        prob_down_k_b_up_kn = dot2(kappa_down_k_b_up_kn_down_bj, nu_up_bj, 4, 1);
        prob_down_mu_alpha_u_k_up_kn = reshape(dot2(ne_pi_down_mu_alpha_u_k_up_b, prob_down_k_b_up_kn, 5, 2), param.n_mu, param.n_alpha, param.n_u, ne_param.n_k, []);
        ne_T_down_mu_alpha_u_k_up_un_kn = permute(reshape(outer(param.phi_down_mu_u_up_un, reshape(permute(prob_down_mu_alpha_u_k_up_kn, [1 3 2 4 5]), param.n_mu, param.n_u, []), 3, 3), param.n_mu, param.n_u, param.n_u, param.n_alpha, ne_param.n_k, []), [1 4 2 5 3 6]);

        % 3.2
        ne_Q_down_mu_alpha_u_k = permute(dot2(permute(ne_pi_down_mu_alpha_u_k_up_b, [1 2 4 3 5]), cost_down_u_b, 5, 2), [1 2 4 3]);

        % 3.3
        num_J_iter = 1;
        J_error = inf;
        while J_error > ne_param.J_tol && num_J_iter <= ne_param.J_max_iter
            for i_alpha = 1 : param.n_alpha
                alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                if alpha == 1
                    if num_ne_pi_iter <= 10 || mod(num_ne_pi_iter, 100) == 0
                        rel_v = ne_Q_down_mu_alpha_u_k(1,1) + dot2(reshape(ne_T_down_mu_alpha_u_k_up_un_kn(1,1,:,:), 1, []), reshape(ne_J_down_mu_alpha_u_k, [], 1), 2, 1);
                        ne_J_down_mu_alpha_u_k_next = ne_Q_down_mu_alpha_u_k + dot2(reshape(ne_T_down_mu_alpha_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_J_down_mu_alpha_u_k, [], 1), 3, 1) - rel_v;
                    else
                        ne_J_down_mu_alpha_u_k_next = ne_J_down_mu_alpha_u_k;
                    end
                else
                    ne_J_down_mu_alpha_u_k_next(:,i_alpha,:,:) = ne_Q_down_mu_alpha_u_k(:,i_alpha,:,:) + alpha * reshape(dot2(permute(reshape(ne_T_down_mu_alpha_u_k_up_un_kn(:,i_alpha,:,:,:,:), param.n_mu, 1, param.n_u, ne_param.n_k, []), [1 2 5 3 4]), reshape(ne_J_down_mu_alpha_u_k(:,i_alpha,:,:), param.n_mu, 1, []), 3, 3), param.n_mu, 1, param.n_u, []);
                end
            end
            J_error = norm(reshape(ne_J_down_mu_alpha_u_k - ne_J_down_mu_alpha_u_k_next, 1, []), inf);
            ne_J_down_mu_alpha_u_k = ne_J_down_mu_alpha_u_k_next;
            num_J_iter = num_J_iter + 1;
        end
        if num_J_iter > ne_param.J_max_iter
            fprintf('\nWARNING: Bellman recursion did not converge! Error: %f\n\n', J_error);
        end
        if alpha == 1 && (num_ne_pi_iter <= 10 || mod(num_ne_pi_iter, 100) == 0)
            ne_J_down_mu_alpha_u_k = ne_Q_down_mu_alpha_u_k + dot2(reshape(ne_T_down_mu_alpha_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_J_down_mu_alpha_u_k, [], 1), 3, 1) - ne_J_down_mu_alpha_u_k;
            J_max_dev = max(ne_J_down_mu_alpha_u_k(:)) - min(ne_J_down_mu_alpha_u_k(:));
            if J_max_dev > ne_param.v_tol
                fprintf('\nWARNING: Identical average stage cost assumption unsatisfied! Max deviation: %f\n\n', J_max_dev);
            end
        end

        %% Step 4
        % 4.1
        cost_down_u_k_b = permute(outer(ones(ne_param.n_k, 1), cost_down_u_b), [2 1 3]);
        cost_down_alpha_u_k_b = reshape(outer(ones(param.n_alpha, 1), cost_down_u_k_b), [param.n_alpha, size(cost_down_u_k_b)]);
        q_down_mu_alpha_u_k_b = reshape(outer(ones(param.n_mu, 1), cost_down_alpha_u_k_b), [param.n_mu, size(cost_down_alpha_u_k_b)]);
        
        prob_down_mu_u_k_b_up_un_kn = permute(reshape(outer(reshape(param.phi_down_mu_u_up_un, [], 1), prob_down_k_b_up_kn), [size(param.phi_down_mu_u_up_un), size(prob_down_k_b_up_kn)]), [1 2 4 5 3 6]);
        t_down_mu_alpha_u_k_b_up_un_kn = permute(reshape(outer(ones(param.n_alpha, 1), prob_down_mu_u_k_b_up_un_kn), [param.n_alpha, size(prob_down_mu_u_k_b_up_un_kn)]), [2 1 3 4 5 6 7]);
        future_cost_down_mu_alpha_u_k_b = reshape(dot2(permute(reshape(t_down_mu_alpha_u_k_b_up_un_kn, param.n_mu, param.n_alpha, param.n_u, ne_param.n_k, ne_param.n_k, []), [1 2 6 3 4 5]), reshape(ne_J_down_mu_alpha_u_k, param.n_mu, param.n_alpha, []), 3, 3), param.n_mu, param.n_alpha, param.n_u, ne_param.n_k, []);
        
        % 4.2
        ne_F_down_mu_alpha_u_k_b = zeros(param.n_mu, param.n_alpha, param.n_u, ne_param.n_k, ne_param.n_k);
        for i_alpha = 1 : param.n_alpha
            alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
            ne_F_down_mu_alpha_u_k_b(:,i_alpha,:,:) = q_down_mu_alpha_u_k_b(:,i_alpha,:,:) + alpha * future_cost_down_mu_alpha_u_k_b(:,i_alpha,:,:);
        end

        % 4.3
        br_pi_down_mu_alpha_u_k_up_b = zeros(param.n_mu, param.n_alpha, param.n_u, ne_param.n_k, ne_param.n_k);
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                for i_u = 1 : param.n_u
                    for i_k = 1 : ne_param.n_k
                        [min_F, i_min_b] = min(ne_F_down_mu_alpha_u_k_b(i_mu,i_alpha,i_u,i_k,1:i_k));
                        if abs((min_F - ne_J_down_mu_alpha_u_k(i_mu,i_alpha,i_u,i_k)) / ne_J_down_mu_alpha_u_k(i_mu,i_alpha,i_u,i_k)) <= ne_param.br_J_tol
                            br_pi_down_mu_alpha_u_k_up_b(i_mu,i_alpha,i_u,i_k,:) = ne_pi_down_mu_alpha_u_k_up_b(i_mu,i_alpha,i_u,i_k,:);
                        else
                            br_pi_down_mu_alpha_u_k_up_b(i_mu,i_alpha,i_u,i_k,i_min_b) = 1;
                        end
                    end
                end
            end
        end
        if ne_param.plot
            % Best response policy plot
            br_pi_plot_fg = 3;
            br_pi_plot_pos = [0, 0, screenwidth, screenheight];
            plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, br_pi_down_mu_alpha_u_k_up_b, param, ne_param, i_alpha_comp);
        end
        
        ne_pi_down_mu_alpha_u_k_up_b_next = (1 - ne_param.ne_pi_mom) * ne_pi_down_mu_alpha_u_k_up_b + ne_param.ne_pi_mom * br_pi_down_mu_alpha_u_k_up_b;
        ne_pi_diff = ne_pi_down_mu_alpha_u_k_up_b_next - ne_pi_down_mu_alpha_u_k_up_b;
        ne_pi_error = norm(reshape(ne_pi_diff, 1, []), inf);
        
        %% Step 2
        % 2.2
        ne_d_up_mu_alpha_u_k_next = reshape(dot2(reshape(ne_T_down_mu_alpha_u_k_up_un_kn, param.n_mu, param.n_alpha, [], param.n_u, ne_param.n_k), reshape(ne_d_up_mu_alpha_u_k, param.n_mu, param.n_alpha, []), 3, 3), param.n_mu, param.n_alpha, param.n_u, []);
        ne_d_up_mu_alpha_u_k_next = ne_d_up_mu_alpha_u_k_next / sum(ne_d_up_mu_alpha_u_k_next(:));
        ne_d_up_mu_alpha_u_k_next = (1 - ne_param.ne_d_mom) * ne_d_up_mu_alpha_u_k + ne_param.ne_d_mom * ne_d_up_mu_alpha_u_k_next;
        prob_up_k_u_alpha = dot2(permute(ne_d_up_mu_alpha_u_k_next, [4 3 2 1]), ones(param.n_mu, 1), 4, 1);
        prob_up_k_u = dot2(prob_up_k_u_alpha, ones(param.n_alpha, 1), 3, 1);
        sigma_up_k = dot2(prob_up_k_u, ones(param.n_u, 1), 2, 1);
        if sigma_up_k(end) > ne_param.max_sigma_k_max
            k_max_saturated = true;
            break;
        end
        k_bar_diff = (param.k_bar - sigma_up_k.' * ne_param.K) / param.k_bar;
        if k_bar_diff ~= 0 && k_bar_diff < sigma_up_k(1)
            for i_mu = 1 : param.n_mu
                for i_alpha = 1 : param.n_alpha
                    for i_u = 1 : param.n_u
                        ne_d_up_mu_alpha_u_k_next(i_mu,i_alpha,i_u,1) = ne_d_up_mu_alpha_u_k_next(i_mu,i_alpha,i_u,1) - param.g_up_mu_alpha(i_mu,i_alpha) * prob_down_mu_up_u_init(i_mu,i_u) * k_bar_diff;
                        ne_d_up_mu_alpha_u_k_next(i_mu,i_alpha,i_u,ne_param.i_k_bar) = ne_d_up_mu_alpha_u_k_next(i_mu,i_alpha,i_u,ne_param.i_k_bar) + param.g_up_mu_alpha(i_mu,i_alpha) * prob_down_mu_up_u_init(i_mu,i_u) * k_bar_diff;
                    end
                end
            end
        end
        ne_d_error = norm(reshape(ne_d_up_mu_alpha_u_k_next - ne_d_up_mu_alpha_u_k, 1, []), inf);

        %% Update NE policy & distribution candidates
        ne_pi_down_mu_alpha_u_k_up_b = ne_pi_down_mu_alpha_u_k_up_b_next;
        ne_d_up_mu_alpha_u_k = ne_d_up_mu_alpha_u_k_next;
        
        % Plot
        if ne_param.plot
            plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_mu_alpha_u_k_up_b, param, ne_param, i_alpha_comp);
            plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_mu_alpha_u_k, param, ne_param, i_alpha_comp);
            drawnow;
        end
        
        % Display status
        fprintf('Iteration %d policy error %f distribution error %f\n', num_ne_pi_iter, ne_pi_error, ne_d_error);
        
        % Update history
        if ne_param.store_hist
            ne_pi_hist(num_ne_pi_iter+1,:) = reshape(ne_pi_down_mu_alpha_u_k_up_b, 1, []);
            ne_d_hist(num_ne_pi_iter+1,:) = reshape(ne_d_up_mu_alpha_u_k, 1, []);
        end
        ne_pi_error_hist(num_ne_pi_iter) = ne_pi_error;
        ne_d_error_hist(num_ne_pi_iter) = ne_d_error;
        
        % Increment iteration count
        num_ne_pi_iter = num_ne_pi_iter + 1;
    end
    
    if k_max_saturated
        ne_param.k_max = ne_param.k_max + ne_param.k_max_step;
        assert(ne_param.k_max <= ne_param.max_k_max, 'Increased k_max too much. Aborting');
        fprintf('\nWARNING: Too many agents saturating. Increased k_max to %02d\n\n', ne_param.k_max);
        ne_param.K = (0 : ne_param.k_max).';
        ne_param.n_k = length(ne_param.K);
        [zeta_down_u_b_bj, kappa_down_k_b_up_kn_down_bj] = get_game_tensors(param, ne_param);
        pi_down_mu_alpha_u_k_up_b_init = get_pi_init(param, ne_param);
        [prob_down_mu_up_u_init, prob_down_mu_alpha_up_k_init, d_up_mu_alpha_u_k_init] = get_d_init(param.k_bar, param, ne_param);
        close all;
        continue;
    end
    
    % Remove 'extra' history
    if ne_param.store_hist
        ne_pi_hist(num_ne_pi_iter+1:end,:) = [];
        ne_d_hist(num_ne_pi_iter+1:end,:) = [];
    end
    ne_pi_error_hist(num_ne_pi_iter:end) = [];
    ne_d_error_hist(num_ne_pi_iter:end) = [];
    
    % Some final results
    prob_up_k_u_alpha = dot2(permute(ne_d_up_mu_alpha_u_k, [4 3 2 1]), ones(param.n_mu, 1), 4, 1);
    prob_up_k_u = dot2(prob_up_k_u_alpha, ones(param.n_alpha, 1), 3, 1);
    ne_upsilon_up_u = dot2(prob_up_k_u, ones(ne_param.n_k, 1), 1, 1).';
    ne_sigma_up_k = dot2(prob_up_k_u, ones(param.n_u, 1), 2, 1);
    ne_sigma_down_mu_alpha_up_k = zeros(param.n_mu, param.n_alpha, ne_param.n_k);
    for i_mu = 1 : param.n_mu
        for i_alpha = 1 : param.n_alpha
            ne_sigma_down_mu_alpha_up_k(i_mu,i_alpha,:) = dot2(squeeze(ne_d_up_mu_alpha_u_k_next(i_mu,i_alpha,:,:)), ones(param.n_u, 1), 1, 1);
            ne_sigma_down_mu_alpha_up_k(i_mu,i_alpha,:) = ne_sigma_down_mu_alpha_up_k(i_mu,i_alpha,:) / sum(ne_sigma_down_mu_alpha_up_k(i_mu,i_alpha,:));
        end
    end
    
    if alpha == 1
        num_J_iter = 1;
        J_error = inf;
        while J_error > ne_param.v_tol && num_J_iter <= ne_param.v_max_iter
            rel_v = ne_Q_down_mu_alpha_u_k(1,1) + dot2(reshape(ne_T_down_mu_alpha_u_k_up_un_kn(1,1,:,:), 1, []), reshape(ne_J_down_mu_alpha_u_k, [], 1), 2, 1);
            ne_J_down_mu_alpha_u_k_next = ne_Q_down_mu_alpha_u_k + dot2(reshape(ne_T_down_mu_alpha_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_J_down_mu_alpha_u_k, [], 1), 3, 1) - rel_v;
            J_error = norm(ne_J_down_mu_alpha_u_k - ne_J_down_mu_alpha_u_k_next, inf);
            ne_J_down_mu_alpha_u_k = ne_J_down_mu_alpha_u_k_next;
            num_J_iter = num_J_iter + 1;
        end
        if num_J_iter > ne_param.v_max_iter
            fprintf('\nWARNING: Value iteration did not converge! Error: %f\n\n', J_error);
        end
        ne_J_down_mu_alpha_u_k = ne_Q_down_mu_alpha_u_k + dot2(reshape(ne_T_down_mu_alpha_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_J_down_mu_alpha_u_k, [], 1), 3, 1) - ne_J_down_mu_alpha_u_k;
        J_max_dev = max(ne_J_down_mu_alpha_u_k(:)) - min(ne_J_down_mu_alpha_u_k(:));
        if J_max_dev > ne_param.v_tol
            fprintf('\nWARNING: Identical average stage cost assumption unsatisfied! Max deviation: %f\n\n', J_max_dev);
        end
    end
    
    % Plot remaining statistics
    if ne_param.plot
        % NE karma distribution plot
        ne_sigma_plot_fg = 4;
        ne_sigma_plot_pos = [0, 0, screenwidth, screenheight];
        plot_ne_sigma(ne_sigma_plot_fg, ne_sigma_plot_pos, ne_sigma_down_mu_alpha_up_k, param, ne_param, i_alpha_comp);
        
        % NE payoffs plot
        ne_J_plot_fg = 5;
        ne_J_plot_pos = [0, 0, screenwidth, screenheight];
        plot_ne_J(ne_J_plot_fg, ne_J_plot_pos, ne_J_down_mu_alpha_u_k, param, ne_param, i_alpha_comp);

        % NE payoffs per bid plot
        ne_F_plot_fg = 6;
        ne_F_plot_pos = [0, 0, screenwidth, screenheight];
        plot_ne_F(ne_F_plot_fg, ne_F_plot_pos, parula, ne_F_down_mu_alpha_u_k_b, param, ne_param, i_alpha_comp);

        % NE state transitions plot
        ne_T_plot_fg = 7;
        ne_T_plot_pos = [0, 0, screenwidth, screenheight];
        plot_ne_T(ne_T_plot_fg, ne_T_plot_pos, RedColormap, ne_T_down_mu_alpha_u_k_up_un_kn, param, ne_param, i_alpha_comp);

        % NE policy error plot
        ne_pi_error_plot_fg = 8;
        ne_pi_error_plot_pos = [0, 0, screenwidth, screenheight];
        plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_pi_error_hist, param, i_alpha_comp);
    end
    
    % Store end results
    if ne_param.save
        file_str = ['karma_nash_equilibrium/results/k_bar_', num2str(param.k_bar, '%02d')];
        if param.n_alpha == 1
            if alpha > 0.99 && alpha < 1
                alpha_str = num2str(alpha, '%.3f');
            else
                alpha_str = num2str(alpha, '%.2f');
            end
            file_str = [file_str, '_alpha_', alpha_str, '.mat'];
        else
            file_str = [file_str, '_z'];
            for i_alpha = 1 : param.n_alpha
                file_str = [file_str, '_', num2str(param.z_up_alpha(i_alpha), '%.2f')];
            end
            file_str = [file_str, '.mat'];
        end
        save(file_str);
    end
    
    i_alpha_comp = i_alpha_comp + 1;
end
i_alpha_comp = i_alpha_comp - 1;

%% If plotting is not active, plot everything at the end
if ~ne_param.plot
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

    % NE karma distribution plot
    ne_sigma_plot_fg = 4;
    ne_sigma_plot_pos = [0, 0, screenwidth, screenheight];
    plot_ne_sigma(ne_sigma_plot_fg, ne_sigma_plot_pos, ne_sigma_down_mu_alpha_up_k, param, ne_param, i_alpha_comp);
    
    % NE payoffs plot
    ne_J_plot_fg = 5;
    ne_J_plot_pos = [0, 0, screenwidth, screenheight];
    plot_ne_J(ne_J_plot_fg, ne_J_plot_pos, ne_J_down_mu_alpha_u_k, param, ne_param, i_alpha_comp);
    
    % NE payoffs per bid plot
    ne_F_plot_fg = 6;
    ne_F_plot_pos = [0, 0, screenwidth, screenheight];
    plot_ne_F(ne_F_plot_fg, ne_F_plot_pos, parula, ne_F_down_mu_alpha_u_k_b, param, ne_param, i_alpha_comp);

    % NE state transitions plot
    ne_T_plot_fg = 7;
    ne_T_plot_pos = [0, 0, screenwidth, screenheight];
    plot_ne_T(ne_T_plot_fg, ne_T_plot_pos, RedColormap, ne_T_down_mu_alpha_u_k_up_un_kn, param, ne_param, i_alpha_comp);

    % NE policy error plot
    ne_pi_error_plot_fg = 8;
    ne_pi_error_plot_pos = [0, 0, screenwidth, screenheight];
    plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_pi_error_hist, param, i_alpha_comp);
end

%% Inform user when done
fprintf('DONE\n\n');