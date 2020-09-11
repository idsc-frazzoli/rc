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

%% Game tensors
[c_down_u_m_mj, kappa_down_k_m_up_kn_down_mj] = ne_func.get_game_tensors(ne_param);

%% Optimization variables
pi = sdpvar(ne_param.num_X, ne_param.num_K, 'full');
d = sdpvar(ne_param.num_X, 1, 'full');
iota = sdpvar(ne_param.num_K, 1, 'full');
t = sdpvar(ne_param.num_X, ne_param.num_X, 'full');

%% Initial guess for first k_ave
k_ave = ne_param.k_ave(1);
i_kave = find(ne_param.K == k_ave);
pi_down_u_k_up_m_init = ne_func.get_pi_init(ne_param);
[p_up_u_init, s_up_k_init, d_up_u_k_init] = ne_func.get_d_init(k_ave, ne_param);
num_d_iter = 1;
d_init_error = inf;
while d_init_error > ne_param.d_tol && num_d_iter <= ne_param.d_max_iter
    iota_up_mj_init = dot2(reshape(permute(pi_down_u_k_up_m_init, [3 1 2]), ne_param.num_K, []), reshape(d_up_u_k_init, [], 1), 2, 1);
    lambda_down_k_m_up_kn = dot2(kappa_down_k_m_up_kn_down_mj, iota_up_mj_init, 4, 1);
    sigma_down_u_k_up_kn = squeeze(dot2(pi_down_u_k_up_m_init, lambda_down_k_m_up_kn, 3, 2));
    t_down_u_k_up_un_kn_init = permute(reshape(outer(ne_param.mu_down_u_up_un, reshape(sigma_down_u_k_up_kn, ne_param.num_U, []), 2, 2), ne_param.num_U, ne_param.num_U, ne_param.num_K, ne_param.num_K), [1 3 2 4]);
    d_up_u_k_init_next = squeeze(dot2(reshape(t_down_u_k_up_un_kn_init, [], ne_param.num_U, ne_param.num_K), reshape(d_up_u_k_init, [], 1), 1, 1));
    d_up_u_k_init_next = d_up_u_k_init_next / sum(d_up_u_k_init_next(:));
    s_up_k = sum(d_up_u_k_init_next);
    assert(s_up_k(end) < ne_param.max_s_k_max, 'Too many agents saturating. Increase k_max.');
    k_ave_diff = (k_ave - s_up_k * ne_param.K) / k_ave;
    if k_ave_diff ~= 0 && k_ave_diff < s_up_k(1)
        for i_u = 1 : ne_param.num_U
            d_up_u_k_init_next(i_u,1) = d_up_u_k_init_next(i_u,1) - p_up_u_init(i_u) * k_ave_diff;
            d_up_u_k_init_next(i_u,i_kave) = d_up_u_k_init_next(i_u,i_kave) + p_up_u_init(i_u) * k_ave_diff;
        end
    end
    d_init_error = norm(d_up_u_k_init_next - d_up_u_k_init, inf);
    d_up_u_k_init = d_up_u_k_init_next;
    num_d_iter = num_d_iter + 1;
end
iota_up_mj_init = dot2(reshape(permute(pi_down_u_k_up_m_init, [3 1 2]), ne_param.num_K, []), reshape(d_up_u_k_init, [], 1), 2, 1);
lambda_down_k_m_up_kn = dot2(kappa_down_k_m_up_kn_down_mj, iota_up_mj_init, 4, 1);
sigma_down_u_k_up_kn = squeeze(dot2(pi_down_u_k_up_m_init, lambda_down_k_m_up_kn, 3, 2));
t_down_u_k_up_un_kn_init = permute(reshape(outer(ne_param.mu_down_u_up_un, reshape(sigma_down_u_k_up_kn, ne_param.num_U, []), 2, 2), ne_param.num_U, ne_param.num_U, ne_param.num_K, ne_param.num_K), [1 3 2 4]);

pi_0 = reshape(permute(pi_down_u_k_up_m_init, [2 1 3]), ne_param.num_X, []);
d_0 = reshape(d_up_u_k_init.', [], 1);
iota_0 = iota_up_mj_init;
t_0 = reshape(permute(t_down_u_k_up_un_kn_init, [2 1 4 3]), ne_param.num_X, []);

assign(pi, pi_0);
assign(d, d_0);
assign(iota, iota_0);
assign(t, t_0);

%% Constraints
constraints = [];
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        if i_u == 1 || i_k == 1
            constraints = [constraints; pi(i_x,1) == 1];
            constraints = [constraints; pi(i_x,2:end) == 0];
        else
            constraints = [constraints; pi(i_x,1) == 0];
            constraints = [constraints; pi(i_x,2:i_k) >= 0];
            constraints = [constraints; sum(pi(i_x,2:i_k)) == 1];
            if i_k < ne_param.num_K
                constraints = [constraints; pi(i_x,i_k+1:end) == 0];
            end
        end
    end
end
constraints = [constraints; d >= 0];
constraints = [constraints; sum(d) == 1];
constraints = [constraints; iota == pi.' * d];
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        for i_un = 1 : ne_param.num_U
            i_un_base = (i_un - 1) * ne_param.num_K;
            for i_kn = 1 : ne_param.num_K
                i_xn = i_un_base + i_kn;
                constraints = [constraints; t(i_x,i_xn) == pi(i_x,:) * ne_param.mu_down_u_up_un(i_u,i_un) * squeeze(kappa_down_k_m_up_kn_down_mj(i_k,:,i_kn,:)) * iota];
            end
        end
    end
end
constraints = [constraints; d <= t.' * d + ne_param.d_tol];
constraints = [constraints; d >= t.' * d - ne_param.d_tol];
constraints = [constraints; repmat(ne_param.K, ne_param.num_U, 1).' * d == k_ave];

%% Objective
objective = d(ne_param.num_K+1);

%% Loop over k_ave
for i_k_ave = 1 : length(ne_param.k_ave)
    k_ave = ne_param.k_ave(i_k_ave);
    i_kave = find(ne_param.K == k_ave);
    fprintf('%%%%K_AVE = %d%%%%\n\n', k_ave);

    %% Update k_ave
    if i_k_ave > 1
        % Initial guess
        [p_up_u_init, s_up_k_init, d_up_u_k_init] = ne_func.get_d_init(k_ave, ne_param);
        num_d_iter = 1;
        d_init_error = inf;
        while d_init_error > ne_param.d_tol && num_d_iter <= ne_param.d_max_iter
            iota_up_mj_init = dot2(reshape(permute(pi_down_u_k_up_m_init, [3 1 2]), ne_param.num_K, []), reshape(d_up_u_k_init, [], 1), 2, 1);
            lambda_down_k_m_up_kn = dot2(kappa_down_k_m_up_kn_down_mj, iota_up_mj_init, 4, 1);
            sigma_down_u_k_up_kn = squeeze(dot2(pi_down_u_k_up_m_init, lambda_down_k_m_up_kn, 3, 2));
            t_down_u_k_up_un_kn_init = permute(reshape(outer(ne_param.mu_down_u_up_un, reshape(sigma_down_u_k_up_kn, ne_param.num_U, []), 2, 2), ne_param.num_U, ne_param.num_U, ne_param.num_K, ne_param.num_K), [1 3 2 4]);
            d_up_u_k_init_next = squeeze(dot2(reshape(t_down_u_k_up_un_kn_init, [], ne_param.num_U, ne_param.num_K), reshape(d_up_u_k_init, [], 1), 1, 1));
            d_up_u_k_init_next = d_up_u_k_init_next / sum(d_up_u_k_init_next(:));
            s_up_k = sum(d_up_u_k_init_next);
            assert(s_up_k(end) < ne_param.max_s_k_max, 'Too many agents saturating. Increase k_max.');
            k_ave_diff = (k_ave - s_up_k * ne_param.K) / k_ave;
            if k_ave_diff ~= 0 && k_ave_diff < s_up_k(1)
                for i_u = 1 : ne_param.num_U
                    d_up_u_k_init_next(i_u,1) = d_up_u_k_init_next(i_u,1) - p_up_u_init(i_u) * k_ave_diff;
                    d_up_u_k_init_next(i_u,i_kave) = d_up_u_k_init_next(i_u,i_kave) + p_up_u_init(i_u) * k_ave_diff;
                end
            end
            d_init_error = norm(d_up_u_k_init_next - d_up_u_k_init, inf);
            d_up_u_k_init = d_up_u_k_init_next;
            num_d_iter = num_d_iter + 1;
        end
        iota_up_mj_init = dot2(reshape(permute(pi_down_u_k_up_m_init, [3 1 2]), ne_param.num_K, []), reshape(d_up_u_k_init, [], 1), 2, 1);
        lambda_down_k_m_up_kn = dot2(kappa_down_k_m_up_kn_down_mj, iota_up_mj_init, 4, 1);
        sigma_down_u_k_up_kn = squeeze(dot2(pi_down_u_k_up_m_init, lambda_down_k_m_up_kn, 3, 2));
        t_down_u_k_up_un_kn_init = permute(reshape(outer(ne_param.mu_down_u_up_un, reshape(sigma_down_u_k_up_kn, ne_param.num_U, []), 2, 2), ne_param.num_U, ne_param.num_U, ne_param.num_K, ne_param.num_K), [1 3 2 4]);

        pi_0 = reshape(permute(pi_down_u_k_up_m_init, [2 1 3]), ne_param.num_X, []);
        d_0 = reshape(d_up_u_k_init.', [], 1);
        iota_0 = iota_up_mj_init;
        t_0 = reshape(permute(t_down_u_k_up_un_kn_init, [2 1 4 3]), ne_param.num_X, []);

        assign(pi, pi_0);
        assign(d, d_0);
        assign(iota, iota_0);
        assign(t, t_0);
        
        % k_ave constraint
        constraints(end) = repmat(ne_param.K, ne_param.num_U, 1).' * d == k_ave;
    end

    %% Solve
    options = sdpsettings('solver', 'ipopt', 'usex0', 1, 'verbose', 3, 'ipopt.max_iter', ne_param.ne_pi_max_iter, 'ipopt.max_cpu_time', 10800);
    diagnostics = optimize(constraints, objective, options);

    %% Get solution
    pi_val = double(pi);
    d_val = double(d);
    b_val = double(iota);
    t_val = double(t);
    obj_val = double(objective);

    %% Convert to tensors as compatible with Nash equilibrium algorithm
    % Get policy in tensor form
    sw_pi_down_u_k_up_m = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
    for i_u = 1 : ne_param.num_U
        i_u_base = (i_u - 1) * ne_param.num_K;
        for i_k = 1 : ne_param.num_K
            i_x = i_u_base + i_k;
            sw_pi_down_u_k_up_m(i_u,i_k,:) = pi_val(i_x,:);
        end
    end
    
    % Get stationary distribution in tensor form
    sw_d_up_u_k = zeros(ne_param.num_U, ne_param.num_K);
    for i_u = 1 : ne_param.num_U
        i_u_base = (i_u - 1) * ne_param.num_K;
        for i_k = 1 : ne_param.num_K
            i_x = i_u_base + i_k;
            sw_d_up_u_k(i_u,i_k) = d_val(i_x);
        end
    end
    
    % Make sure stationary distribution is indeed stationary. The following
    % logic should terminate in 1 iteration
    num_d_iter = 1;
    sw_d_error = inf;
    while sw_d_error > ne_param.d_tol && num_d_iter <= ne_param.d_max_iter
        iota_up_mj = dot2(reshape(permute(sw_pi_down_u_k_up_m, [3 1 2]), ne_param.num_K, []), reshape(sw_d_up_u_k, [], 1), 2, 1);
        lambda_down_k_m_up_kn = dot2(kappa_down_k_m_up_kn_down_mj, iota_up_mj, 4, 1);
        sigma_down_u_k_up_kn = squeeze(dot2(sw_pi_down_u_k_up_m, lambda_down_k_m_up_kn, 3, 2));
        sw_t_down_u_k_up_un_kn = permute(reshape(outer(ne_param.mu_down_u_up_un, reshape(sigma_down_u_k_up_kn, ne_param.num_U, []), 2, 2), ne_param.num_U, ne_param.num_U, ne_param.num_K, ne_param.num_K), [1 3 2 4]);
        sw_d_up_u_k_next = squeeze(dot2(reshape(sw_t_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(sw_d_up_u_k, [], 1), 1, 1));
        sw_d_up_u_k_next = sw_d_up_u_k_next / sum(sw_d_up_u_k_next(:));
        s_up_k = sum(sw_d_up_u_k_next);
        assert(s_up_k(end) < ne_param.max_s_k_max, 'Too many agents saturating. Increase k_max.');
        k_ave_diff = (k_ave - s_up_k * ne_param.K) / k_ave;
        if k_ave_diff ~= 0 && k_ave_diff < s_up_k(1)
            for i_u = 1 : ne_param.num_U
                sw_d_up_u_k_next(i_u,1) = sw_d_up_u_k_next(i_u,1) - p_up_u_init(i_u) * k_ave_diff;
                sw_d_up_u_k_next(i_u,i_kave) = sw_d_up_u_k_next(i_u,i_kave) + p_up_u_init(i_u) * k_ave_diff;
            end
        end
        sw_d_error = norm(sw_d_up_u_k_next - sw_d_up_u_k, inf);
        sw_d_up_u_k = sw_d_up_u_k_next;
        num_d_iter = num_d_iter + 1;
    end
    sw_p_up_u = sum(sw_d_up_u_k, 2);
    sw_s_up_k = sum(sw_d_up_u_k, 1).';
    
    % Get expected stage cost using same logic as NE algorithm
    xi_down_u_m = dot2(c_down_u_m_mj, iota_up_mj, 3, 1);
    sw_q_down_u_k = permute(dot2(permute(sw_pi_down_u_k_up_m, [2 1 3]), xi_down_u_m, 3, 2), [2 1]);

    %% Some sanity checks
    % Confirm stationary distribution of optimization and tensor logic is the
    % same
    d_diff = abs(d_val - reshape(sw_d_up_u_k.', [], 1));
    d_error = max(d_diff)

    % Confirm state transition matrix of optimization and tensor logic is the
    % same
    sw_t_mat = reshape(permute(sw_t_down_u_k_up_un_kn, [2 1 4 3]), ne_param.num_X, []);
    t_diff = abs(t_val - sw_t_mat);
    t_error = max(t_diff(:))

    % Calculate efficiency based on tensors and optimization
    e = -dot(reshape(sw_d_up_u_k, [], 1), reshape(sw_q_down_u_k, [], 1))

    %% Plots
    % SW policy plot
    if i_k_ave == 1
        sw_pi_plot_fg = 1;
        sw_pi_plot_pos = [0, default_height, default_width, default_height];
    end
    ne_func.plot_sw_pi(sw_pi_plot_fg, sw_pi_plot_pos, RedColormap, sw_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.K, k_ave);

    % SW stationary distribution plot
    if i_k_ave == 1
        sw_d_plot_fg = 2;
        sw_d_plot_pos = [0, 0, default_width, default_height];
    end
    ne_func.plot_sw_d(sw_d_plot_fg, sw_d_plot_pos, sw_d_up_u_k, ne_param.U, ne_param.K, k_ave);

    % SW expected stage cost plot
    if i_k_ave == 1
        sw_q_plot_fg = 3;
        sw_q_plot_pos = [0, 0, default_width, default_height];
    end
    ne_func.plot_sw_q(sw_q_plot_fg, sw_q_plot_pos, sw_q_down_u_k, ne_param.U, ne_param.K, k_ave);

    % SW state transitions plot
    if i_k_ave == 1
        sw_t_plot_fg = 4;
        sw_t_plot_pos = [0, 0, screenwidth, screenheight];
    end
    ne_func.plot_sw_t(sw_t_plot_fg, sw_t_plot_pos, RedColormap, sw_t_down_u_k_up_un_kn, ne_param.U, ne_param.K, k_ave);
    
    %% Store end results
    if ne_param.save
        save(['karma_nash_equilibrium/results/k_ave_', num2str(k_ave, '%02d'), '.mat']);
    end
end