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
% Probability of winning/losing given messages
gamma_down_m_mj_up_o = zeros(ne_param.num_M, ne_param.num_M, ne_param.num_O);
for i_m = 1 : ne_param.num_M
    m = ne_param.M(i_m);
    for i_mj = 1 : ne_param.num_M
        mj = ne_param.M(i_mj);
        switch ne_param.smoothing
            case 1
                gamma_down_m_mj_up_o(i_m,i_mj,1) = 0.5 * (1 + erf((m - mj) / (sqrt(2) * ne_param.gamma_s)));
            case 2
                gamma_down_m_mj_up_o(i_m,i_mj,1) = 1 / (1 + exp(-(m - mj) / ne_param.gamma_s));
            otherwise
                gamma_down_m_mj_up_o(i_m,i_mj,1) = max([0, min([(m - mj + 1) / 2, 1])]);
        end
        gamma_down_m_mj_up_o(i_m,i_mj,2) = 1 - gamma_down_m_mj_up_o(i_m,i_mj,1);
    end
end

% Game cost tensor
zeta_down_u_o = outer(ne_param.U, ne_param.O);
c_down_u_m_mj = squeeze(dot2(zeta_down_u_o, permute(gamma_down_m_mj_up_o, [3 1 2]), 2, 1));
clearvars zeta_down_u_o;

% Game state transition tensor
beta_down_k_m_up_mb = zeros(ne_param.num_K, ne_param.num_M, ne_param.num_K);
for i_k = 1 : ne_param.num_K
    k = ne_param.K(i_k);
    for i_m = 1 : ne_param.num_M
        m = ne_param.M(i_m);
        if m > k
            continue
        end
        for i_mb = 1 : i_k
            mb = ne_param.K(i_mb);
            switch ne_param.smoothing
                case 1
                    beta_down_k_m_up_mb(i_k,i_m,i_mb) = exp(-(mb - m)^2 / (2 * ne_param.beta_s^2));
                case 2
                    beta_down_k_m_up_mb(i_k,i_m,i_mb) = exp(-(mb - m) / ne_param.beta_s) / ((1 + exp(-(mb - m) / ne_param.beta_s))^2);
                otherwise
                    beta_down_k_m_up_mb(i_k,i_m,i_mb) = max([0, 1 - abs(mb - m)]);
            end  
        end
        beta_down_k_m_up_mb(i_k,i_m,:) = beta_down_k_m_up_mb(i_k,i_m,:) / sum(beta_down_k_m_up_mb(i_k,i_m,:));
    end
end
epsilon_down_k_mb_kj_mbj_o_up_kn = ...
    zeros(ne_param.num_K, ne_param.num_K, ne_param.num_K, ne_param.num_K, ne_param.num_O, ne_param.num_K);
for i_k = 1 : ne_param.num_K
    k = ne_param.K(i_k);
    for i_mb = 1 : i_k
        mb = ne_param.K(i_mb);
        for i_kj = 1 : ne_param.num_K
            kj = ne_param.K(i_kj);
            for i_mbj = 1 : i_kj
                mbj = ne_param.K(i_mbj);
                
                i_kn_win = ne_param.K == k - min([mb, ne_param.k_max - kj]);
                epsilon_down_k_mb_kj_mbj_o_up_kn(i_k,i_mb,i_kj,i_mbj,1,i_kn_win) = 1;
                
                i_kn_lose = ne_param.K == min([k + mbj, ne_param.k_max]);
                epsilon_down_k_mb_kj_mbj_o_up_kn(i_k,i_mb,i_kj,i_mbj,2,i_kn_lose) = 1;
            end
        end
    end
end
if ne_param.m_interval ~= 1 || ne_param.smoothing ~= 0
    upsilon_down_k_mb_kj_mj_o_up_kn = permute(squeeze(dot2(permute(epsilon_down_k_mb_kj_mbj_o_up_kn, [1 2 5 6 3 4]), permute(beta_down_k_m_up_mb, [1 3 2]), 6, 2)), [1 2 5 6 3 4]);
    clearvars epsilon_down_k_mb_kj_mbj_o_up_kn;
    phi_down_k_m_kj_mj_o_up_kn = permute(squeeze(dot2(permute(upsilon_down_k_mb_kj_mj_o_up_kn, [3 4 5 6 1 2]), permute(beta_down_k_m_up_mb, [1 3 2]), 6, 2)), [5 6 1 2 3 4]);
    % phi_down_k_m_kj_mj_o_up_kn = zeros(ne_param.num_K, ne_param.num_M, ne_param.num_K, ne_param.num_M, ne_param.num_O, ne_param.num_K);
    % parfor i_k = 1 : ne_param.num_K
    %     v = zeros(ne_param.num_M, ne_param.num_K, ne_param.num_M, ne_param.num_O, ne_param.num_K);
    %     k = ne_param.K(i_k);
    %     for i_m = 1 : ne_param.num_M
    %         if ne_param.M(i_m) > k
    %             continue;
    %         end
    %         for i_kj = 1 : ne_param.num_K
    %             kj = ne_param.K(i_kj);
    %             for i_mj = 1 : ne_param.num_M
    %                 if ne_param.M(i_mj) > kj
    %                     continue;
    %                 end
    %                 for i_o = 1 : ne_param.num_O
    %                     for i_kn = 1 : ne_param.num_K
    %                         v(i_m,i_kj,i_mj,i_o,i_kn) = dot(squeeze(upsilon_down_k_mb_kj_mj_o_up_kn(i_k,:,i_kj,i_mj,i_o,i_kn)), squeeze(beta_down_k_m_up_mb(i_k,i_m,:)));
    %                     end
    %                 end
    %             end
    %         end
    %     end
    %     phi_down_k_m_kj_mj_o_up_kn(i_k,:,:,:,:,:) = v;
    % end
    clearvars beta_down_k_m_up_mb upsilon_down_k_mb_kj_mj_o_up_kn;
else
    phi_down_k_m_kj_mj_o_up_kn = epsilon_down_k_mb_kj_mbj_o_up_kn;
    clearvars beta_down_k_m_up_mb epsilon_down_k_mb_kj_mbj_o_up_kn;
end
kappa_down_k_m_kj_mj_up_kn = permute(dot2(permute(phi_down_k_m_kj_mj_o_up_kn, [1 3 6 2 4 5]), gamma_down_m_mj_up_o, 6, 3), [1 4 2 5 3]);
clearvars gamma_down_m_mj_up_o phi_down_k_m_kj_mj_o_up_kn;
psi_down_u_k_m_kj_mj_up_un_kn = permute(reshape(outer(reshape(ne_param.mu_down_u_up_un, [], 1), kappa_down_k_m_kj_mj_up_kn), [ne_param.num_U, ne_param.num_U, size(kappa_down_k_m_kj_mj_up_kn)]), [1 3 4 5 6 2 7]);
clearvars kappa_down_k_m_kj_mj_up_kn;

%% From game tensors to cells of 2 dimensional matrices
Psi = cell(ne_param.num_X);
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        for i_un = 1 : ne_param.num_U
            i_un_base = (i_un - 1) * ne_param.num_K;
            for i_kn = 1 : ne_param.num_K
                i_xn = i_un_base + i_kn;
                Psi{i_x,i_xn} = zeros(ne_param.num_M, ne_param.num_X * ne_param.num_M);
                for i_uj = 1 : ne_param.num_U
                    i_uj_base = (i_uj - 1) * ne_param.num_K;
                    for i_kj = 1 : ne_param.num_K
                        i_xj = i_uj_base + i_kj;
                        i_xj_base = (i_xj - 1) * ne_param.num_M;
                        for i_mj = 1 : ne_param.num_M
                            i_xj_mj = i_xj_base + i_mj;
                            Psi{i_x,i_xn}(:,i_xj_mj) = squeeze(psi_down_u_k_m_kj_mj_up_un_kn(i_u,i_k,:,i_kj,i_mj,i_un,i_kn));
                        end
                    end
                end
            end
        end
    end
end

%% Optimization variables
pi = sdpvar(ne_param.num_X, ne_param.num_M, 'full');
d = sdpvar(ne_param.num_X, 1, 'full');
b = sdpvar(ne_param.num_X * ne_param.num_M, 1, 'full');
t = sdpvar(ne_param.num_X, ne_param.num_X, 'full');

%% Constraints
constraints = [];
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        k = ne_param.K(i_k);
        if ne_param.U(i_u) == 0 || k == 0
            constraints = [constraints; pi(i_x,1) == 1];
            constraints = [constraints; pi(i_x,2:end) == 0];
        else
            i_adm_m = find(ne_param.M <= ne_param.K(i_k));
            constraints = [constraints; pi(i_x,i_adm_m) >= 0];
            constraints = [constraints; sum(pi(i_x,i_adm_m)) == 1];
            if length(i_adm_m) < ne_param.num_M
                constraints = [constraints; pi(i_x,i_adm_m(end)+1:end) == 0];
            end
        end
    end
end
constraints = [constraints; d >= 0];
constraints = [constraints; sum(d) == 1];
for i_x = 1 : ne_param.num_X
    start_i = (i_x - 1) * ne_param.num_M + 1;
    end_i = i_x * ne_param.num_M;
    constraints = [constraints; b(start_i:end_i) == d(i_x) * pi(i_x,:).'];
end
for i_x = 1 : ne_param.num_X
    for i_xn = 1 : ne_param.num_X
        constraints = [constraints; t(i_x,i_xn) == pi(i_x,:) * Psi{i_x,i_xn} * b];
    end
end
constraints = [constraints; d == t.' * d];

%% Objective
objective = pi(ne_param.num_K+1:end,1).' * d(ne_param.num_K+1:end);

%% Loop over k_ave
for i_k_ave = 1 : length(ne_param.k_ave)
    k_ave = ne_param.k_ave(i_k_ave);
    fprintf('%%%%K_AVE = %d%%%%\n\n', k_ave);

    %% k_ave constraint
    if i_k_ave == 1
        constraints = [constraints; repmat(ne_param.K, ne_param.num_U, 1).' * d == k_ave];
    else
        constraints(end) = repmat(ne_param.K, ne_param.num_U, 1).' * d == k_ave;
    end
    
    %% Initial guess
    % Policy
    pi_0 = zeros(ne_param.num_X, ne_param.num_M);
    for i_u = 1 : ne_param.num_U
        i_u_base = (i_u - 1) * ne_param.num_K;
        for i_k = 1 : ne_param.num_K
            i_x = i_u_base + i_k;
            k = ne_param.K(i_k);
            if ne_param.U(i_u) == 0 || k == 0
                pi_0(i_x,1) = 1;
            else
                % Bid 1 if urgent
                i_m = find(ne_param.M == 1);
                pi_0(i_x,i_m) = 1;
    %             % Bid all if urgent
    %             i_m = find(ne_param.M == k);
    %             pi_0(i_x,i_m) = 1;
    %             % Bid random if urgent
    %             i_adm_m = find(ne_param.M <= k);
    %             p = 1 / length(i_adm_m);
    %             pi_0(i_x,i_adm_m) = p;
%                 % Bid 1 up to k_ave, then 1 + k - k_ave from there on
%                 if k <= k_ave
%                     i_m = find(ne_param.M == 1);
%                     pi_0(i_x,i_m) = 1;
%                 else
%                     i_m = find(ne_param.M == min([1 + k - k_ave, k]));
%                     pi_0(i_x,i_m) = 1;
%                 end
            end
        end
    end

    % Find initial distribution corresponding to initial policy and with
    % required k_ave
    d_u_0 = ne_func.stat_dist(ne_param.mu_down_u_up_un);
    i_kave = find(ne_param.K == k_ave);
    if k_ave * 2 <= ne_param.k_max
        i_kave2 = find(ne_param.K == k_ave * 2);
        d_k_0 = [1 / i_kave2 * ones(i_kave2, 1); zeros(ne_param.num_K - i_kave2, 1)];
    elseif k_ave >= ne_param.k_max
        d_k_0 = zeros(ne_param.num_K, 1);
        d_k_0(end) = 1;
    else
        d_k_0 = 1 / ne_param.num_K * ones(ne_param.num_K, 1);
        K_small = 0 : k_ave - 1;
        K_big = k_ave + 1 : ne_param.k_max;
        num_K_small = length(K_small);
        num_K_big = length(K_big);
        delta_constant = sum(K_small) / num_K_small - sum(K_big) / num_K_big;
        delta_k_ave = k_ave - ne_param.K.' * d_k_0;
        delta_p = delta_k_ave / delta_constant;
        d_k_0(1:i_kave-1) = d_k_0(1:i_kave-1) + delta_p / num_K_small;
        d_k_0(i_kave+1:end) = d_k_0(i_kave+1:end) - delta_p / num_K_big;
    end
    d_0 = zeros(ne_param.num_X, 1);
    for i_u = 1 : ne_param.num_U
        i_u_base = (i_u - 1) * ne_param.num_K;
        d_0(i_u_base+1:i_u_base+ne_param.num_K) = d_u_0(i_u) * d_k_0;
    end
    d_0 = d_0 / sum(d_0);

    % Get inital guess of b
    b_0 = zeros(ne_param.num_X * ne_param.num_M, 1);
    for i_x = 1 : ne_param.num_X
        start_i = (i_x - 1) * ne_param.num_M + 1;
        end_i = i_x * ne_param.num_M;
        b_0(start_i:end_i) = d_0(i_x) * pi_0(i_x,:).';
    end

    % Get initial guess of t
    t_0 = zeros(ne_param.num_X);
    for i_x = 1 : ne_param.num_X
        for i_xn = 1 : ne_param.num_X
            t_0(i_x,i_xn) = pi_0(i_x,:) * Psi{i_x,i_xn} * b_0;
        end
    end

    % Update initial guesses d_0, b_0, t_0 until convergence
    while norm(d_0 - t_0.' * d_0, inf) > 1e-12
        d_0 = t_0.' * d_0;
        d_0 = d_0 / sum(d_0);

        for i_x = 1 : ne_param.num_X
            start_i = (i_x - 1) * ne_param.num_M + 1;
            end_i = i_x * ne_param.num_M;
            b_0(start_i:end_i) = d_0(i_x) * pi_0(i_x,:).';
        end

        for i_x = 1 : ne_param.num_X
            for i_xn = 1 : ne_param.num_X
                t_0(i_x,i_xn) = pi_0(i_x,:) * Psi{i_x,i_xn} * b_0;
            end
        end
    end

    assign(pi, pi_0);
    assign(d, d_0);
    assign(b, b_0);
    assign(t, t_0);

    %% Solve
    options = sdpsettings('solver', 'ipopt', 'usex0', 1, 'verbose', 3);
    diagnostics = optimize(constraints, objective, options);

    %% Get solution
    pi_val = double(pi);
    d_val = double(d);
    b_val = double(b);
    t_val = double(t);
    obj_val = double(objective);

    %% Convert to tensors as compatible with Nash equilibrium algorithm
    % Get policy in tensor form
    sw_pi_down_u_k_up_m = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_M);
    for i_u = 1 : ne_param.num_U
        i_u_base = (i_u - 1) * ne_param.num_K;
        for i_k = 1 : ne_param.num_K
            i_x = i_u_base + i_k;
            sw_pi_down_u_k_up_m(i_u,i_k,:) = pi_val(i_x,:);
        end
    end

    % Get stationary distribution from policy using same logic as NE algorithm
    lambda_down_u_k_m_uj_kj_up_un_kn = permute(squeeze(dot2(sw_pi_down_u_k_up_m, permute(psi_down_u_k_m_kj_mj_up_un_kn, [4 5 1 2 3 6 7]), 3, 2)), [3 4 5 1 2 6 7]);
    sigma_down_u_k_uj_kj_up_un_kn = squeeze(dot2(sw_pi_down_u_k_up_m, lambda_down_u_k_m_uj_kj_up_un_kn, 3, 3));
    sw_d_up_u_k = outer(d_u_0, d_k_0);
    sw_t_down_u_k_up_un_kn = squeeze(dot2(reshape(sw_d_up_u_k, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
    sw_d_up_u_k_next = squeeze(dot2(reshape(sw_t_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(sw_d_up_u_k, [], 1), 1, 1));
    sw_d_up_u_k_next = sw_d_up_u_k_next / sum(sw_d_up_u_k_next(:));
    sw_d_error = norm(sw_d_up_u_k - sw_d_up_u_k_next, inf);
    sw_d_up_u_k = sw_d_up_u_k_next;
    num_d_iter = 0;
    while sw_d_error > 1e-12 && num_d_iter < 1000
        sw_t_down_u_k_up_un_kn = squeeze(dot2(reshape(sw_d_up_u_k, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
        sw_d_up_u_k_next = squeeze(dot2(reshape(sw_t_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(sw_d_up_u_k, [], 1), 1, 1));
        sw_d_up_u_k_next = sw_d_up_u_k_next / sum(sw_d_up_u_k_next(:));
        sw_d_error = norm(sw_d_up_u_k - sw_d_up_u_k_next, inf);
        sw_d_up_u_k = sw_d_up_u_k_next;
        num_d_iter = num_d_iter + 1;
    end

    % Get expected stage cost using same logic as NE algorithm
    iota_up_mj = squeeze(dot2(reshape(sw_d_up_u_k, [], 1), reshape(sw_pi_down_u_k_up_m, [], ne_param.num_M), 1, 1));
    xi_down_u_m = dot2(iota_up_mj, c_down_u_m_mj, 2, 3);
    sw_q_down_u_k = squeeze(dot2(xi_down_u_m, permute(sw_pi_down_u_k_up_m, [1 3 2]), 2, 2));

    %% Some sanity checks
    % Confirm stationary distribution of optimization and tensor logic is the
    % same
    d_diff = abs(d_val - reshape(sw_d_up_u_k.', [], 1));
    d_error = max(d_diff)

    % Confirm state transition matrix of optimization and tensor logic is the
    % same
    sw_t_mat = zeros(ne_param.num_X);
    for i_u = 1 : ne_param.num_U
        start_i_u = (i_u - 1) * ne_param.num_K + 1;
        end_i_u = i_u * ne_param.num_K;
        for i_un = 1 : ne_param.num_U
            start_i_un = (i_un - 1) * ne_param.num_K + 1;
            end_i_un = i_un * ne_param.num_K;
            sw_t_mat(start_i_u:end_i_u,start_i_un:end_i_un) = squeeze(sw_t_down_u_k_up_un_kn(i_u,:,i_un,:));
        end
    end
    t_diff = abs(t_val - sw_t_mat);
    t_error = max(t_diff(:))

    % Calculate efficiency based on tensors and optimization
    e = dot(reshape(sw_d_up_u_k, [], 1), reshape(sw_q_down_u_k, [], 1))
    p_up_u = sum(sw_d_up_u_k, 2);
    e_opt = 0.5 * (obj_val * p_up_u(1) + p_up_u(2)^2) * ne_param.U(end);
    e_error = abs(e - e_opt)

    %% Plots
    % SW policy plot
    if i_k_ave == 1
        sw_pi_plot_fg = 1;
        sw_pi_plot_pos = [0, default_height, default_width, default_height];
    end
    ne_func.plot_sw_pi(sw_pi_plot_fg, sw_pi_plot_pos, RedColormap, sw_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, k_ave);

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