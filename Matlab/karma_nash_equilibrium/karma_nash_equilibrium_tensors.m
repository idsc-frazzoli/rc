clear;
close all;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
default_width = screenwidth / 2;
default_height = screenheight / 3;
load('karma_nash_equilibrium/RedColormap.mat');

%% Parameters
% NE computation parameters
ne_param = load_ne_parameters();

%% Iterative algorithm to find Karma Nash equilibrium %%

%% Step 0: 
D_up_u = ne_param.p_U;
gamma_down_mi_mj_up_oi = zeros(ne_param.num_K, ne_param.num_K, ne_param.num_O);
for i_mi = 1 : ne_param.num_K
    for i_mj = 1 : ne_param.num_K
        if i_mi > i_mj  % Agent i wins
            gamma_down_mi_mj_up_oi(i_mi,i_mj,1) = 1;
        elseif i_mi < i_mj % Agent j wins
            gamma_down_mi_mj_up_oi(i_mi,i_mj,2) = 1;
        else % Tie
            gamma_down_mi_mj_up_oi(i_mi,i_mj,1) = 0.5;
            gamma_down_mi_mj_up_oi(i_mi,i_mj,2) = 0.5;
        end
    end
end
c_down_ui_oi = ne_param.U * ne_param.O.';
phi_down_ki_mi_kj_mj_oi_up_kin...
    = zeros(ne_param.num_K, ne_param.num_K, ne_param.num_K, ne_param.num_K, ne_param.num_O, ne_param.num_K);
for i_ki = 1 : ne_param.num_K
    ki = ne_param.K(i_ki);
    for i_mi = 1 : i_ki
        mi = ne_param.K(i_mi);
        for i_kj = 1 : ne_param.num_K
            kj = ne_param.K(i_kj);
            for i_mj = 1 : i_kj
                mj = ne_param.K(i_mj);
                
                % Next karma if agent i wins
                kin = ki - min([mi, ne_param.k_max - kj]);
                i_kin = ne_param.K == kin;
                phi_down_ki_mi_kj_mj_oi_up_kin(i_ki,i_mi,i_kj,i_mj,1,i_kin) = 1;
                
                % Next karma if agent i loses
                kin = min([ki + mj, ne_param.k_max]);
                i_kin = ne_param.K == kin;
                phi_down_ki_mi_kj_mj_oi_up_kin(i_ki,i_mi,i_kj,i_mj,2,i_kin) = 1;
            end
        end
    end
end
phi_down_ki_mi_kj_mj_oi_up_uin_kin = permute(outer(phi_down_ki_mi_kj_mj_oi_up_kin, D_up_u, 6, 1), [1 2 3 4 5 7 6]);

%% Step 1
pi_down_u_k_up_m = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
for i_k = 1 : ne_param.num_K
    % Initial policy for nonurgent is to always bid 0
    pi_down_u_k_up_m(1,i_k,1) = 1;
    % Initial policy for urgent is uniform distribution over messages
    p = 1 / i_k;
    for i_m = 1 : i_k
        pi_down_u_k_up_m(2,i_k,i_m) = p;
    end
end

% Plot
policy_plot_fg = 1;
policy_plot_pos = [0, 2 * default_height, default_width, default_height];
policy_plot_title = 'Current NE Guess Policy';
ne_func.plot_policy_tensors(policy_plot_fg, policy_plot_pos, pi_down_u_k_up_m, ne_param, policy_plot_title, RedColormap);

%% Step 2
% 2.1
T_down_ki_mi_kj_mj_up_uin_kin = zeros(ne_param.num_K, ne_param.num_K, ne_param.num_K, ne_param.num_K, ne_param.num_U, ne_param.num_K);
for i_ki = 1 : ne_param.num_K
    for i_kj = 1 : ne_param.num_K
        for i_uin = 1 : ne_param.num_U
            for i_kin = 1 : ne_param.num_K
                T_down_ki_mi_kj_mj_up_uin_kin(i_ki,:,i_kj,:,i_uin,i_kin) = dot2(gamma_down_mi_mj_up_oi, squeeze(phi_down_ki_mi_kj_mj_oi_up_uin_kin(i_ki,:,i_kj,:,:,i_uin,i_kin)), 3, 3);
            end
        end
    end
end
T_down_ki_mi_uj_kj_up_uin_kin = zeros(ne_param.num_K, ne_param.num_K, ne_param.num_U, ne_param.num_K, ne_param.num_U, ne_param.num_K);
for i_ki = 1 : ne_param.num_K
    for i_mi = 1 : ne_param.num_K
        for i_uj = 1 : ne_param.num_U
            for i_uin = 1 : ne_param.num_U
                for i_kin = 1 : ne_param.num_K
                    T_down_ki_mi_uj_kj_up_uin_kin(i_ki,i_mi,i_uj,:,i_uin,i_kin) = dot2(squeeze(pi_down_u_k_up_m(i_uj,:,:)), squeeze(T_down_ki_mi_kj_mj_up_uin_kin(i_ki,i_mi,:,:,i_uin,i_kin)), 2, 2);
                end
            end
        end
    end
end
T_down_ui_ki_uj_kj_up_uin_kin = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_U, ne_param.num_K, ne_param.num_U, ne_param.num_K);
for i_ui = 1 : ne_param.num_U
    for i_uj = 1 : ne_param.num_U
        for i_kj = 1 : ne_param.num_K
            for i_uin = 1 : ne_param.num_U
                for i_kin = 1 : ne_param.num_K
                    T_down_ui_ki_uj_kj_up_uin_kin(i_ui,:,i_uj,i_kj,i_uin,i_kin) = dot2(squeeze(pi_down_u_k_up_m(i_ui,:,:)), squeeze(T_down_ki_mi_uj_kj_up_uin_kin(:,:,i_uj,i_kj,i_uin,i_kin)), 2, 2);
                end
            end
        end
    end
end

% 2.2
% D_up_u_k = D_up_u * 1 / ne_param.num_K * ones(1, ne_param.num_K);
D_up_u_k = D_up_u * 1 / (floor(ne_param.num_K / 4)) * [ones(1, floor(ne_param.num_K / 4)), zeros(1, ne_param.num_K - floor(ne_param.num_K / 4))];
% D_up_u_k = zeros(ne_param.num_U, ne_param.num_K);
% D_up_u_k(:,7) = D_up_u;
%D_up_u_k = 1 / (ne_param.num_U * ne_param.num_K) * ones(ne_param.num_U, ne_param.num_K);
%D_up_u_k = rand(ne_param.num_U, ne_param.num_K);
D_up_u_k = D_up_u_k / sum(D_up_u_k(:));
T_down_ui_ki_up_uin_kin = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(T_down_ui_ki_uj_kj_up_uin_kin, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
D_up_u_k_next = squeeze(dot2(reshape(T_down_ui_ki_up_uin_kin, [], ne_param.num_U, ne_param.num_K), reshape(D_up_u_k, [], 1), 1, 1));
D_up_u_k_next = D_up_u_k_next / sum(D_up_u_k_next(:));
D_error = norm(D_up_u_k - D_up_u_k_next, inf);
D_up_u_k = D_up_u_k_next;
num_D_iter = 0;
while D_error > ne_param.D_tol && num_D_iter < ne_param.D_max_iter
    T_down_ui_ki_up_uin_kin = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(T_down_ui_ki_uj_kj_up_uin_kin, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
    D_up_u_k_next = squeeze(dot2(reshape(T_down_ui_ki_up_uin_kin, [], ne_param.num_U, ne_param.num_K), reshape(D_up_u_k, [], 1), 1, 1));
    D_up_u_k_next = D_up_u_k_next / sum(D_up_u_k_next(:));
    D_error = norm(D_up_u_k - D_up_u_k_next, inf);
    D_up_u_k = D_up_u_k_next;
    num_D_iter = num_D_iter + 1;
end

% Plot
D_plot_fg = 2;
D_plot_pos = [0, default_height / 3, default_width, default_height];
D_plot_title = 'Current NE Stationary Distribution';
ne_func.plot_D_tensors(D_plot_fg, D_plot_pos, D_up_u_k, ne_param, D_plot_title);

%% Step 3
% 3.1
c_down_ui_mi_mj = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
for i_ui = 1 : ne_param.num_U
    c_down_ui_mi_mj(i_ui,:,:) = dot2(gamma_down_mi_mj_up_oi, c_down_ui_oi(i_ui,:), 3, 2);
end
D_up_mj = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(pi_down_u_k_up_m, [], ne_param.num_K), 1, 1));
q_down_ui_ki_mi = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
for i_ki = 1 : ne_param.num_K
    q_down_ui_ki_mi(:,i_ki,:) = dot2(D_up_mj, c_down_ui_mi_mj, 2, 3);
end
T_down_ui_ki_mi_up_uin_kin = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K, ne_param.num_U, ne_param.num_K);
for i_ui = 1 : ne_param.num_U
    T_down_ui_ki_mi_up_uin_kin(i_ui,:,:,:,:) = dot2(reshape(D_up_u_k, [], 1), reshape(T_down_ki_mi_uj_kj_up_uin_kin, ne_param.num_K, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3);
end
q_down_ui_ki = dot2(pi_down_u_k_up_m, q_down_ui_ki_mi, 3, 3);

% 3.2
V_down_ui_ki = zeros(ne_param.num_U, ne_param.num_K);
V_down_ui_ki_next = q_down_ui_ki + ne_param.alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
V_down_ui_ki = V_down_ui_ki_next;
num_V_iter = 0;
while V_error > ne_param.V_tol && num_V_iter < ne_param.V_max_iter
    V_down_ui_ki_next = q_down_ui_ki + ne_param.alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
    V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
    V_down_ui_ki = V_down_ui_ki_next;
    num_V_iter = num_V_iter + 1;
end

% Plot
V_plot_fg = 3;
V_plot_pos = [default_width, default_height / 3, default_width, default_height];
V_plot_title = 'Current NE Expected Utility';
ne_func.plot_V_tensors(V_plot_fg, V_plot_pos, V_down_ui_ki, ne_param, V_plot_title);

%% Step 4
% 4.1
V_down_ui_ki_mi = q_down_ui_ki_mi + ne_param.alpha * dot2(reshape(T_down_ui_ki_mi_up_uin_kin, ne_param.num_U, ne_param.num_K, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 4, 1);

% 4.2
pi_i_down_ui_ki_up_mi = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
policy_error = 0;
for i_ui = 1 : ne_param.num_U
    for i_ki = 1 : ne_param.num_K
        [~, i_mi] = min(V_down_ui_ki_mi(i_ui,i_ki,1:i_ki));
        pi_i_down_ui_ki_up_mi(i_ui,i_ki,i_mi) = 1;
        policy_error = max([policy_error,...
            norm(squeeze(pi_down_u_k_up_m(i_ui,i_ki,:) - pi_i_down_ui_ki_up_mi(i_ui,i_ki,:)), inf)]);
    end
end

% 4.3/4
policy_i_error = policy_error;
num_ne_iter = 0;
num_policy_iter = 0;
% Display status
fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_ne_iter, num_policy_iter, policy_i_error);
while policy_i_error > ne_param.policy_tol && num_policy_iter < ne_param.policy_max_iter
    % 4.4.1
    q_down_ui_ki = dot2(pi_i_down_ui_ki_up_mi, q_down_ui_ki_mi, 3, 3);
    T_down_ui_ki_up_uin_kin = dot2(pi_i_down_ui_ki_up_mi, T_down_ui_ki_mi_up_uin_kin, 3, 3);
    
    % 4.4.2
    V_down_ui_ki = zeros(ne_param.num_U, ne_param.num_K);
    V_down_ui_ki_next = q_down_ui_ki + ne_param.alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
    V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
    V_down_ui_ki = V_down_ui_ki_next;
    num_V_iter = 0;
    while V_error > ne_param.V_tol && num_V_iter < ne_param.V_max_iter
        V_down_ui_ki_next = q_down_ui_ki + ne_param.alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
        V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
        V_down_ui_ki = V_down_ui_ki_next;
        num_V_iter = num_V_iter + 1;
    end
    
    % 4.4.3
    V_down_ui_ki_mi = q_down_ui_ki_mi + ne_param.alpha * dot2(reshape(T_down_ui_ki_mi_up_uin_kin, ne_param.num_U, ne_param.num_K, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 4, 1);
    
    % 4.4.4
    pi_i_down_ui_ki_up_mi_next = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
    policy_i_error = 0;
    for i_ui = 1 : ne_param.num_U
        for i_ki = 1 : ne_param.num_K
            [~, i_mi] = min(V_down_ui_ki_mi(i_ui,i_ki,1:i_ki));
            pi_i_down_ui_ki_up_mi_next(i_ui,i_ki,i_mi) = 1;
            policy_i_error = max([policy_i_error,...
                norm(squeeze(pi_i_down_ui_ki_up_mi(i_ui,i_ki,:) - pi_i_down_ui_ki_up_mi_next(i_ui,i_ki,:)), inf)]);
        end
    end
    
    % 4.4.5
    pi_i_down_ui_ki_up_mi = pi_i_down_ui_ki_up_mi_next;
    num_policy_iter = num_policy_iter + 1;
    % Display status
    fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_ne_iter, num_policy_iter, policy_i_error);
end

% Plot
policy_i_plot_fg = 4;
policy_i_plot_pos = [default_width, 2 * default_height, default_width, default_height];
policy_i_plot_title = 'Best Response Policy';
ne_func.plot_policy_tensors(policy_i_plot_fg, policy_i_plot_pos, pi_i_down_ui_ki_up_mi, ne_param, policy_i_plot_title, RedColormap);


%% Step 5
pi_down_u_k_up_m = pi_i_down_ui_ki_up_mi;
% Plot
if ne_param.plot
    ne_func.plot_policy_tensors(policy_plot_fg, policy_plot_pos, pi_down_u_k_up_m, ne_param, policy_plot_title, RedColormap);
end
% Display status and store history of policies
fprintf('Iteration %d policy error %f\n', num_ne_iter, policy_error);
policy_hist_end = zeros(1, ne_param.num_X);
for i_ui = 1 : ne_param.num_U
    base_i_ui = (i_ui - 1) * ne_param.num_K;
    for i_ki = 1 : ne_param.num_K
        i_xi = base_i_ui + i_ki;
        [~, max_i] = max(pi_down_u_k_up_m(i_ui,i_ki,:));
        policy_hist_end(i_xi) = ne_param.K(max_i);
        if i_xi == 1
            fprintf('Iteration %d policy:\t%d', num_ne_iter, policy_hist_end(i_xi));
        elseif mod(i_xi - 1, ne_param.num_K) == 0
            fprintf('\n\t\t\t\t\t%d', policy_hist_end(i_xi));
        else
            fprintf('->%d', policy_hist_end(i_xi));
        end
    end
end
policy_hist = policy_hist_end;
fprintf('\n\n');
while policy_error > ne_param.policy_tol && num_ne_iter < ne_param.ne_policy_max_iter
    %% Step 5.2
    % 5.2.1
    for i_ki = 1 : ne_param.num_K
        for i_mi = 1 : ne_param.num_K
            for i_uj = 1 : ne_param.num_U
                for i_uin = 1 : ne_param.num_U
                    for i_kin = 1 : ne_param.num_K
                        T_down_ki_mi_uj_kj_up_uin_kin(i_ki,i_mi,i_uj,:,i_uin,i_kin) = dot2(squeeze(pi_down_u_k_up_m(i_uj,:,:)), squeeze(T_down_ki_mi_kj_mj_up_uin_kin(i_ki,i_mi,:,:,i_uin,i_kin)), 2, 2);
                    end
                end
            end
        end
    end
    for i_ui = 1 : ne_param.num_U
        for i_uj = 1 : ne_param.num_U
            for i_kj = 1 : ne_param.num_K
                for i_uin = 1 : ne_param.num_U
                    for i_kin = 1 : ne_param.num_K
                        T_down_ui_ki_uj_kj_up_uin_kin(i_ui,:,i_uj,i_kj,i_uin,i_kin) = dot2(squeeze(pi_down_u_k_up_m(i_ui,:,:)), squeeze(T_down_ki_mi_uj_kj_up_uin_kin(:,:,i_uj,i_kj,i_uin,i_kin)), 2, 2);
                    end
                end
            end
        end
    end

    % 5.2.2
%     D_up_u_k = D_up_u * 1 / ne_param.num_K * ones(1, ne_param.num_K);
    D_up_u_k = D_up_u * 1 / (floor(ne_param.num_K / 4)) * [ones(1, floor(ne_param.num_K / 4)), zeros(1, ne_param.num_K - floor(ne_param.num_K / 4))];
%     D_up_u_k = zeros(ne_param.num_U, ne_param.num_K);
%     D_up_u_k(:,7) = D_up_u;
    %D_up_u_k = 1 / (ne_param.num_U * ne_param.num_K) * ones(ne_param.num_U, ne_param.num_K);
    %D_up_u_k = rand(ne_param.num_U, ne_param.num_K);
    D_up_u_k = D_up_u_k / sum(D_up_u_k(:));
    T_down_ui_ki_up_uin_kin = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(T_down_ui_ki_uj_kj_up_uin_kin, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
    D_up_u_k_next = squeeze(dot2(reshape(T_down_ui_ki_up_uin_kin, [], ne_param.num_U, ne_param.num_K), reshape(D_up_u_k, [], 1), 1, 1));
    D_up_u_k_next = D_up_u_k_next / sum(D_up_u_k_next(:));
    D_error = norm(D_up_u_k - D_up_u_k_next, inf);
    D_up_u_k = D_up_u_k_next;
    num_D_iter = 0;
    while D_error > ne_param.D_tol && num_D_iter < ne_param.D_max_iter
        T_down_ui_ki_up_uin_kin = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(T_down_ui_ki_uj_kj_up_uin_kin, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
        D_up_u_k_next = squeeze(dot2(reshape(T_down_ui_ki_up_uin_kin, [], ne_param.num_U, ne_param.num_K), reshape(D_up_u_k, [], 1), 1, 1));
        D_up_u_k_next = D_up_u_k_next / sum(D_up_u_k_next(:));
        D_error = norm(D_up_u_k - D_up_u_k_next, inf);
        D_up_u_k = D_up_u_k_next;
        num_D_iter = num_D_iter + 1;
    end

    % Plot
    if ne_param.plot
        ne_func.plot_D_tensors(D_plot_fg, D_plot_pos, D_up_u_k, ne_param, D_plot_title);
    end

    %% Step 5.3
    % 5.3.1
    D_up_mj = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(pi_down_u_k_up_m, [], ne_param.num_K), 1, 1));
    for i_ki = 1 : ne_param.num_K
        q_down_ui_ki_mi(:,i_ki,:) = dot2(D_up_mj, c_down_ui_mi_mj, 2, 3);
    end
    for i_ui = 1 : ne_param.num_U
        T_down_ui_ki_mi_up_uin_kin(i_ui,:,:,:,:) = dot2(reshape(D_up_u_k, [], 1), reshape(T_down_ki_mi_uj_kj_up_uin_kin, ne_param.num_K, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3);
    end
    q_down_ui_ki = dot2(pi_down_u_k_up_m, q_down_ui_ki_mi, 3, 3);

    % 5.3.2
    V_down_ui_ki = zeros(ne_param.num_U, ne_param.num_K);
    V_down_ui_ki_next = q_down_ui_ki + ne_param.alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
    V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
    V_down_ui_ki = V_down_ui_ki_next;
    num_V_iter = 0;
    while V_error > ne_param.V_tol && num_V_iter < ne_param.V_max_iter
        V_down_ui_ki_next = q_down_ui_ki + ne_param.alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
        V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
        V_down_ui_ki = V_down_ui_ki_next;
        num_V_iter = num_V_iter + 1;
    end

    % Plot
    if ne_param.plot
        ne_func.plot_V_tensors(V_plot_fg, V_plot_pos, V_down_ui_ki, ne_param, V_plot_title);
    end

    %% Step 5.4
    % 5.4.1
    V_down_ui_ki_mi = q_down_ui_ki_mi + ne_param.alpha * dot2(reshape(T_down_ui_ki_mi_up_uin_kin, ne_param.num_U, ne_param.num_K, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 4, 1);

    % 5.4.2
    pi_i_down_ui_ki_up_mi = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
    policy_error = 0;
    for i_ui = 1 : ne_param.num_U
        for i_ki = 1 : ne_param.num_K
            [~, i_mi] = min(V_down_ui_ki_mi(i_ui,i_ki,1:i_ki));
            pi_i_down_ui_ki_up_mi(i_ui,i_ki,i_mi) = 1;
            policy_error = max([policy_error,...
                norm(squeeze(pi_down_u_k_up_m(i_ui,i_ki,:) - pi_i_down_ui_ki_up_mi(i_ui,i_ki,:)), inf)]);
        end
    end

    % 5.4.3/4
    policy_i_error = policy_error;
    num_policy_iter = 0;
    % Display status
    fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_ne_iter, num_policy_iter, policy_i_error);
    while policy_i_error > ne_param.policy_tol && num_policy_iter < ne_param.policy_max_iter
        % 5.4.4.1
        q_down_ui_ki = dot2(pi_i_down_ui_ki_up_mi, q_down_ui_ki_mi, 3, 3);
        T_down_ui_ki_up_uin_kin = dot2(pi_i_down_ui_ki_up_mi, T_down_ui_ki_mi_up_uin_kin, 3, 3);

        % 5.4.4.2
        V_down_ui_ki = zeros(ne_param.num_U, ne_param.num_K);
        V_down_ui_ki_next = q_down_ui_ki + ne_param.alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
        V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
        V_down_ui_ki = V_down_ui_ki_next;
        num_V_iter = 0;
        while V_error > ne_param.V_tol && num_V_iter < ne_param.V_max_iter
            V_down_ui_ki_next = q_down_ui_ki + ne_param.alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
            V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
            V_down_ui_ki = V_down_ui_ki_next;
            num_V_iter = num_V_iter + 1;
        end

        % 5.4.4.3
        V_down_ui_ki_mi = q_down_ui_ki_mi + ne_param.alpha * dot2(reshape(T_down_ui_ki_mi_up_uin_kin, ne_param.num_U, ne_param.num_K, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 4, 1);

        % 5.4.4.4
        pi_i_down_ui_ki_up_mi_next = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
        policy_i_error = 0;
        for i_ui = 1 : ne_param.num_U
            for i_ki = 1 : ne_param.num_K
                [~, i_mi] = min(V_down_ui_ki_mi(i_ui,i_ki,1:i_ki));
                pi_i_down_ui_ki_up_mi_next(i_ui,i_ki,i_mi) = 1;
                policy_i_error = max([policy_i_error,...
                    norm(squeeze(pi_i_down_ui_ki_up_mi(i_ui,i_ki,:) - pi_i_down_ui_ki_up_mi_next(i_ui,i_ki,:)), inf)]);
            end
        end

        % 5.4.4.5
        pi_i_down_ui_ki_up_mi = pi_i_down_ui_ki_up_mi_next;
        num_policy_iter = num_policy_iter + 1;
        % Display status
        fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_ne_iter, num_policy_iter, policy_i_error);
    end

    % Plot
    if ne_param.plot
        ne_func.plot_policy_tensors(policy_i_plot_fg, policy_i_plot_pos, pi_i_down_ui_ki_up_mi, ne_param, policy_i_plot_title, RedColormap);
    end
    
    %% Step 5.5
    pi_down_u_k_up_m = pi_i_down_ui_ki_up_mi;
    % Plot
    if ne_param.plot
        ne_func.plot_policy_tensors(policy_plot_fg, policy_plot_pos, pi_down_u_k_up_m, ne_param, policy_plot_title, RedColormap);
    end
    num_ne_iter = num_ne_iter + 1;
    % Display status and store history of policies
    fprintf('Iteration %d policy error %f\n', num_ne_iter, policy_error);
    for i_ui = 1 : ne_param.num_U
        base_i_ui = (i_ui - 1) * ne_param.num_K;
        for i_ki = 1 : ne_param.num_K
            i_xi = base_i_ui + i_ki;
            [~, max_i] = max(pi_down_u_k_up_m(i_ui,i_ki,:));
            policy_hist_end(i_xi) = ne_param.K(max_i);
            if i_xi == 1
                fprintf('Iteration %d policy:\t%d', num_ne_iter, policy_hist_end(i_xi));
            elseif mod(i_xi - 1, ne_param.num_K) == 0
                fprintf('\n\t\t\t\t\t%d', policy_hist_end(i_xi));
            else
                fprintf('->%d', policy_hist_end(i_xi));
            end
        end
    end
    policy_hist = [policy_hist; policy_hist_end];
    fprintf('\n\n');
end

if ~ne_param.plot
    ne_func.plot_policy_tensors(policy_plot_fg, policy_plot_pos, pi_down_u_k_up_m, ne_param, policy_plot_title, RedColormap);
    ne_func.plot_D_tensors(D_plot_fg, D_plot_pos, D_up_u_k, ne_param, D_plot_title);
    ne_func.plot_V_tensors(V_plot_fg, V_plot_pos, V_down_ui_ki, ne_param, V_plot_title);
    ne_func.plot_policy_tensors(policy_i_plot_fg, policy_i_plot_pos, pi_i_down_ui_ki_up_mi, ne_param, policy_i_plot_title, RedColormap);
end

%% Inform user when done
fprintf('DONE\n\n');