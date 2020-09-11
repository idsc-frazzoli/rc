clear;
close all;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
default_width = screenwidth / 2;
default_height = screenheight / 2;
load('karma_nash_equilibrium/RedColormap.mat');

%% Parameters
% NE computation parameters
ne_param = load_ne_parameters();

%% Iterative algorithm to find Karma Nash equilibrium %%
%% Step 0: Game tensors
[c_down_u_m_mj, kappa_down_k_m_up_kn_down_mj] = ne_func.get_game_tensors(ne_param);

% Initial policy
pi_down_u_k_up_m_init = ne_func.get_pi_init(ne_param);

% Plot
if ne_param.plot
    ne_pi_plot_fg = 1;
    ne_pi_plot_pos = [0, default_height, default_width, default_height];
    ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, pi_down_u_k_up_m_init, ne_param.U, ne_param.K, ne_param.K, ne_param.k_ave, ne_param.alpha(1));
end

% Initial distribution
[p_up_u_init, s_up_k_init, d_up_u_k_init] = ne_func.get_d_init(ne_param.k_ave, ne_param);

% Plot
if ne_param.plot
    ne_d_plot_fg = 2;
    ne_d_plot_pos = [0, 0, default_width, default_height];
    ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, d_up_u_k_init, ne_param.U, ne_param.K, ne_param.k_ave, ne_param.alpha(1));
end

%% Loop over alphas
i_alpha = 1;
while i_alpha <= ne_param.num_alpha
    alpha = ne_param.alpha(i_alpha);
    fprintf('%%%%ALPHA = %f%%%%\n\n', alpha);

    %% Step 1: NE policy & stationary distribution guess
    % 1.1
    ne_pi_down_u_k_up_m = pi_down_u_k_up_m_init;
    ne_d_up_u_k = d_up_u_k_init;
    if ne_param.store_hist
        ne_pi_hist = zeros(ne_param.ne_pi_max_iter + 1, ne_param.num_U * ne_param.num_K * ne_param.num_K);
        ne_pi_hist(1,:) = reshape(ne_pi_down_u_k_up_m, 1, []);
        ne_d_hist = zeros(ne_param.ne_pi_max_iter + 1, ne_param.num_U * ne_param.num_K);
        ne_d_hist(1,:) = reshape(ne_d_up_u_k, 1, []);
    end
    
    % Plot
    if ne_param.plot
        ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.K, ne_param.k_ave, alpha);
    end
    
    num_ne_pi_iter = 1;
    ne_pi_error = inf;
    ne_pi_error_hist = zeros(ne_param.ne_pi_max_iter, 1);
    ne_d_error = inf;
    ne_d_error_hist = zeros(ne_param.ne_pi_max_iter, 1);
    ne_v_down_u_k = zeros(ne_param.num_U, ne_param.num_K);
    i_k_ave = find(ne_param.K == ne_param.k_ave);
    saturated = false;
    while (ne_pi_error > ne_param.ne_pi_tol || ne_d_error > ne_param.d_tol) && num_ne_pi_iter <= ne_param.ne_pi_max_iter
        %% Step 3
        % 3.1
        iota_up_mj = dot2(reshape(permute(ne_pi_down_u_k_up_m, [3 1 2]), ne_param.num_K, []), reshape(ne_d_up_u_k, [], 1), 2, 1);
        xi_down_u_m = dot2(c_down_u_m_mj, iota_up_mj, 3, 1);
        lambda_down_k_m_up_kn = dot2(kappa_down_k_m_up_kn_down_mj, iota_up_mj, 4, 1);
        sigma_down_u_k_up_kn = squeeze(dot2(ne_pi_down_u_k_up_m, lambda_down_k_m_up_kn, 3, 2));
        ne_t_down_u_k_up_un_kn = permute(reshape(outer(ne_param.mu_down_u_up_un, reshape(sigma_down_u_k_up_kn, ne_param.num_U, []), 2, 2), ne_param.num_U, ne_param.num_U, ne_param.num_K, ne_param.num_K), [1 3 2 4]);

        % 3.2
        ne_q_down_u_k = permute(dot2(permute(ne_pi_down_u_k_up_m, [2 1 3]), xi_down_u_m, 3, 2), [2 1]);

        % 3.3
        num_v_iter = 1;
        v_error = inf;
        while v_error > ne_param.v_tol && num_v_iter <= ne_param.v_max_iter
            if alpha == 1
                if num_ne_pi_iter <= 10 || mod(num_ne_pi_iter, 100) == 0
                    rel_v = ne_q_down_u_k(1,1) + dot2(reshape(ne_t_down_u_k_up_un_kn(1,1,:,:), 1, []), reshape(ne_v_down_u_k, [], 1), 2, 1);
                    ne_v_down_u_k_next = ne_q_down_u_k + dot2(reshape(ne_t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 3, 1) - rel_v;
                else
                    ne_v_down_u_k_next = ne_v_down_u_k;
                end
            else
                ne_v_down_u_k_next = ne_q_down_u_k + alpha * dot2(reshape(ne_t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 3, 1);
            end
            v_error = norm(ne_v_down_u_k - ne_v_down_u_k_next, inf);
            ne_v_down_u_k = ne_v_down_u_k_next;
            num_v_iter = num_v_iter + 1;
        end
        if num_v_iter > ne_param.v_max_iter
            fprintf('\nWARNING: Value iteration did not converge! Error: %f\n\n', v_error);
        end
        if alpha == 1 && (num_ne_pi_iter <= 10 || mod(num_ne_pi_iter, 100) == 0)
            ne_J_down_u_k = ne_q_down_u_k + dot2(reshape(ne_t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 3, 1) - ne_v_down_u_k;
            J_max_dev = max(ne_J_down_u_k(:)) - min(ne_J_down_u_k(:));
            if J_max_dev > ne_param.v_tol
                fprintf('\nWARNING: Identical average stage cost assumption unsatisfied! Max deviation: %f\n\n', J_max_dev);
            end
        end

        %% Step 4
        % 4.1
        chi_down_u_k_m_up_un_kn = permute(reshape(outer(reshape(ne_param.mu_down_u_up_un, [], 1), lambda_down_k_m_up_kn), [ne_param.num_U, ne_param.num_U, size(lambda_down_k_m_up_kn)]), [1 3 4 2 5]);
        omega_down_u_k_m = dot2(reshape(chi_down_u_k_m_up_un_kn, ne_param.num_U, ne_param.num_K, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 4, 1);
        eta_down_u_k_m = permute(outer(ones(ne_param.num_K, 1), xi_down_u_m), [2 1 3]);

        % 4.2
        ne_rho_down_u_k_m = eta_down_u_k_m + alpha * omega_down_u_k_m;

        % 4.3
        br_pi_down_u_k_up_m = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
        for i_u = 1 : ne_param.num_U
            for i_k = 1 : ne_param.num_K
                [min_rho, i_min_m] = min(ne_rho_down_u_k_m(i_u,i_k,1:i_k));
                if abs((min_rho - ne_v_down_u_k(i_u,i_k)) / ne_v_down_u_k(i_u,i_k)) <= ne_param.br_v_tol
                    br_pi_down_u_k_up_m(i_u,i_k,:) = ne_pi_down_u_k_up_m(i_u,i_k,:);
                else
                    br_pi_down_u_k_up_m(i_u,i_k,i_min_m) = 1;
                end
            end
        end
        if ne_param.plot
            % Best response policy plot
            br_pi_plot_fg = 3;
            br_pi_plot_pos = [default_width, default_height, default_width, default_height];
            ne_func.plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, br_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.K, ne_param.k_ave, alpha);
        end
        
        ne_pi_down_u_k_up_m_next = (1 - ne_param.ne_pi_mom(num_ne_pi_iter)) * ne_pi_down_u_k_up_m + ne_param.ne_pi_mom(num_ne_pi_iter) * br_pi_down_u_k_up_m;
        ne_pi_diff = ne_pi_down_u_k_up_m_next - ne_pi_down_u_k_up_m;
        ne_pi_error = norm(reshape(ne_pi_diff, [], 1), inf);
        
        %% Step 2
        % 2.2
        ne_d_up_u_k_next = squeeze(dot2(reshape(ne_t_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(ne_d_up_u_k, [], 1), 1, 1));
        ne_d_up_u_k_next = ne_d_up_u_k_next / sum(ne_d_up_u_k_next(:));
        ne_d_up_u_k_next = (1 - ne_param.d_mom) * ne_d_up_u_k + ne_param.d_mom * ne_d_up_u_k_next;
        s_up_k = sum(ne_d_up_u_k_next);
        if s_up_k(end) > ne_param.max_s_k_max
            saturated = true;
            break;
        end
        assert(s_up_k(end) < ne_param.max_s_k_max, 'Too many agents saturating. Increase k_max.');
        k_ave_diff = (ne_param.k_ave - s_up_k * ne_param.K) / ne_param.k_ave;
        if k_ave_diff ~= 0 && k_ave_diff < s_up_k(1)
            for i_u = 1 : ne_param.num_U
                ne_d_up_u_k_next(i_u,1) = ne_d_up_u_k_next(i_u,1) - p_up_u_init(i_u) * k_ave_diff;
                ne_d_up_u_k_next(i_u,i_k_ave) = ne_d_up_u_k_next(i_u,i_k_ave) + p_up_u_init(i_u) * k_ave_diff;
            end
        end
        ne_d_error = norm(ne_d_up_u_k_next - ne_d_up_u_k, inf);

        %% Update NE policy & stationary distribution candidates
        ne_pi_down_u_k_up_m = ne_pi_down_u_k_up_m_next;
        ne_d_up_u_k = ne_d_up_u_k_next;
        
        % Plot
        if ne_param.plot
            ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.K, ne_param.k_ave, alpha);
            ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
            drawnow;
        end
        
        % Display status
        fprintf('Iteration %d policy error %f distribution error %f\n', num_ne_pi_iter, ne_pi_error, ne_d_error);
        
        % Update history
        if ne_param.store_hist
            ne_pi_hist(num_ne_pi_iter+1,:) = reshape(ne_pi_down_u_k_up_m, 1, []);
            ne_d_hist(num_ne_pi_iter+1,:) = reshape(ne_d_up_u_k, 1, []);
        end
        ne_pi_error_hist(num_ne_pi_iter) = ne_pi_error;
        ne_d_error_hist(num_ne_pi_iter) = ne_d_error;
        
        % Increment iteration count
        num_ne_pi_iter = num_ne_pi_iter + 1;
    end
    
    if saturated
        ne_param.k_max = ne_param.k_max + ne_param.k_max_step;
        assert(ne_param.k_max <= ne_param.max_k_max, 'Increased k_max too much. Aborting');
        fprintf('\nWARNING: Too many agents saturating. Increased k_max to %02d\n\n', ne_param.k_max);
        ne_param.K = (0 : ne_param.k_max).';
        ne_param.num_K = length(ne_param.K);
        ne_param.num_X = ne_param.num_U * ne_param.num_K;
        [c_down_u_m_mj, kappa_down_k_m_up_kn_down_mj] = ne_func.get_game_tensors(ne_param);
        pi_down_u_k_up_m_init = ne_func.get_pi_init(ne_param);
        [p_up_u_init, s_up_k_init, d_up_u_k_init] = ne_func.get_d_init(ne_param.k_ave, ne_param);
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
    ne_p_up_u = sum(ne_d_up_u_k, 2);
    ne_s_up_k = sum(ne_d_up_u_k).';
    
    if alpha == 1
        num_v_iter = 1;
        v_error = inf;
        while v_error > ne_param.v_tol && num_v_iter <= ne_param.v_max_iter
            rel_v = ne_q_down_u_k(1,1) + dot2(reshape(ne_t_down_u_k_up_un_kn(1,1,:,:), 1, []), reshape(ne_v_down_u_k, [], 1), 2, 1);
            ne_v_down_u_k_next = ne_q_down_u_k + dot2(reshape(ne_t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 3, 1) - rel_v;
            v_error = norm(ne_v_down_u_k - ne_v_down_u_k_next, inf);
            ne_v_down_u_k = ne_v_down_u_k_next;
            num_v_iter = num_v_iter + 1;
        end
        if num_v_iter > ne_param.v_max_iter
            fprintf('\nWARNING: Value iteration did not converge! Error: %f\n\n', v_error);
        end
        ne_J_down_u_k = ne_q_down_u_k + dot2(reshape(ne_t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 3, 1) - ne_v_down_u_k;
        J_max_dev = max(ne_J_down_u_k(:)) - min(ne_J_down_u_k(:));
        if J_max_dev > ne_param.v_tol
            fprintf('\nWARNING: Identical average stage cost assumption unsatisfied! Max deviation: %f\n\n', J_max_dev);
        end
    end
    
    % Plot remaining statistics
    if ne_param.plot
        % NE expected utility plot
        ne_v_plot_fg = 4;
        ne_v_plot_pos = [0, 0, default_width, default_height];
        if alpha == 1
            ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_J_down_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
        else
            ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_v_down_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
        end
        
        % NE expected utiliy per message plot
        ne_rho_plot_fg = 5;
        ne_rho_plot_pos = [default_width, 0, default_width, default_height];
        ne_func.plot_ne_rho(ne_rho_plot_fg, ne_rho_plot_pos, parula, ne_rho_down_u_k_m, ne_param.U, ne_param.K, ne_param.K, ne_param.k_ave, alpha);
        
        % NE state transitions plot
        ne_t_plot_fg = 6;
        ne_t_plot_pos = [0, 0, screenwidth, screenheight];
        ne_func.plot_ne_t(ne_t_plot_fg, ne_t_plot_pos, RedColormap, ne_t_down_u_k_up_un_kn, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
        
        % NE policy error plot
        ne_pi_error_plot_fg = 7;
        ne_pi_error_plot_pos = [default_width, 0, default_width, default_height];
        ne_func.plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_pi_error_hist, ne_param.k_ave, alpha);
    end
    
    % Store end results
    if ne_param.save
        if alpha > 0.99 && alpha < 1
            save(['karma_nash_equilibrium/results/k_ave_', num2str(ne_param.k_ave, '%02d'), '_alpha_', num2str(alpha, '%.3f'), '.mat']);
        else
            save(['karma_nash_equilibrium/results/k_ave_', num2str(ne_param.k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.mat']);
        end
    end
    
    i_alpha = i_alpha + 1;
end

%% If plotting is not active, plot everything at the end
if ~ne_param.plot
    % NE policy plot
    ne_pi_plot_fg = 1;
    ne_pi_plot_pos = [0, default_height, default_width, default_height];
    ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.K, ne_param.k_ave, alpha);
    
    % NE stationary distribution plot
    ne_d_plot_fg = 2;
    ne_d_plot_pos = [0, 0, default_width, default_height];
    ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
    
    % Best response policy plot
    br_pi_plot_fg = 3;
    br_pi_plot_pos = [default_width, default_height, default_width, default_height];
    ne_func.plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, br_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.K, ne_param.k_ave, alpha);

    % NE expected utility plot
    ne_v_plot_fg = 4;
    ne_v_plot_pos = [0, 0, default_width, default_height];
    if alpha == 1
        ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_J_down_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
    else
        ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_v_down_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
    end
    
    % NE expected utiliy per message plot
    ne_rho_plot_fg = 5;
    ne_rho_plot_pos = [default_width, 0, default_width, default_height];
    ne_func.plot_ne_rho(ne_rho_plot_fg, ne_rho_plot_pos, parula, ne_rho_down_u_k_m, ne_param.U, ne_param.K, ne_param.K, ne_param.k_ave, alpha);

    % NE state transitions plot
    ne_t_plot_fg = 6;
    ne_t_plot_pos = [0, 0, screenwidth, screenheight];
    ne_func.plot_ne_t(ne_t_plot_fg, ne_t_plot_pos, RedColormap, ne_t_down_u_k_up_un_kn, ne_param.U, ne_param.K, ne_param.k_ave, alpha);

    % NE policy error plot
    ne_pi_error_plot_fg = 7;
    ne_pi_error_plot_pos = [default_width, 0, default_width, default_height];
    ne_func.plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_pi_error_hist, ne_param.k_ave, alpha);
end

%% Inform user when done
fprintf('DONE\n\n');