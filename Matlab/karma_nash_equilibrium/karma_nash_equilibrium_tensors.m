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
%% Step 0:
D_up_u = ne_param.p_U;
c_down_ui_mi_mj = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
for i_ui = 1 : ne_param.num_U
    for i_mi = 1 : ne_param.num_K
        for i_mj = 1 : ne_param.num_K
            if i_mi > i_mj
                % Agent i wins for sure
                c_down_ui_mi_mj(i_ui,i_mi,i_mj) = 0;
            elseif i_mi < i_mj
                % Agent i loses for sure
                c_down_ui_mi_mj(i_ui,i_mi,i_mj) = ne_param.U(i_ui);
            else
                % 50-50 chances of agent i winning or losing
                c_down_ui_mi_mj(i_ui,i_mi,i_mj) = 0.5 * ne_param.U(i_ui);
            end
        end
    end
end
phi_down_ki_mi_kj_mj_up_kin...
    = zeros(ne_param.num_K, ne_param.num_K, ne_param.num_K, ne_param.num_K, ne_param.num_K);
for i_ki = 1 : ne_param.num_K
    ki = ne_param.K(i_ki);
    for i_mi = 1 : i_ki
        mi = ne_param.K(i_mi);
        for i_kj = 1 : ne_param.num_K
            kj = ne_param.K(i_kj);
            for i_mj = 1 : i_kj
                mj = ne_param.K(i_mj);

                if mi > mj
                    % Agent i wins for sure
                    kin = ki - min([mi, ne_param.k_max - kj]);
                    i_kin = ne_param.K == kin;
                    phi_down_ki_mi_kj_mj_up_kin(i_ki,i_mi,i_kj,i_mj,i_kin) = 1;
                elseif mi < mj
                    % Agent i loses for sure
                    kin = min([ki + mj, ne_param.k_max]);
                    i_kin = ne_param.K == kin;
                    phi_down_ki_mi_kj_mj_up_kin(i_ki,i_mi,i_kj,i_mj,i_kin) = 1;
                else
                    % 50-50 chances of agent i winning or losing
                    kin_win = ki - min([mi, ne_param.k_max - kj]);
                    i_kin_win = ne_param.K == kin_win;
                    kin_lose = min([ki + mj, ne_param.k_max]);
                    if kin_win == kin_lose
                        phi_down_ki_mi_kj_mj_up_kin(i_ki,i_mi,i_kj,i_mj,i_kin_win) = 1;
                    else
                        i_kin_lose = ne_param.K == kin_lose;
                        phi_down_ki_mi_kj_mj_up_kin(i_ki,i_mi,i_kj,i_mj,i_kin_win) = 0.5;
                        phi_down_ki_mi_kj_mj_up_kin(i_ki,i_mi,i_kj,i_mj,i_kin_lose) = 0.5;
                    end
                end
            end
        end
    end
end
phi_down_ki_mi_kj_mj_up_uin_kin = permute(outer(D_up_u, phi_down_ki_mi_kj_mj_up_kin), [2 3 4 5 1 6]);

for i_alpha = 1 : length(ne_param.alpha)
    alpha = ne_param.alpha(i_alpha);
    fprintf('%%%%ALPHA = %f%%%%\n\n', alpha);

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
    if ne_param.plot
        ne_pi_plot_fg = 1;
        ne_pi_plot_pos = [0, default_height, default_width, default_height];
        ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, pi_down_u_k_up_m, ne_param.U, ne_param.K, alpha);
    end

    %% Step 2
    % 2.1
    T_down_ki_mi_uj_kj_up_uin_kin = permute(squeeze(dot2(permute(phi_down_ki_mi_kj_mj_up_uin_kin, [1 2 5 6 3 4]), permute(pi_down_u_k_up_m, [2 3 1]), 6, 2)), [1 2 6 5 3 4]);
    T_down_ui_ki_uj_kj_up_uin_kin = permute(squeeze(dot2(permute(T_down_ki_mi_uj_kj_up_uin_kin, [3 4 5 6 1 2]), permute(pi_down_u_k_up_m, [2 3 1]), 6, 2)), [6 5 1 2 3 4]);

    % 2.2
    i_kave = find(ne_param.K == ne_param.k_ave);
    % D_up_k = zeros(ne_param.num_K, 1);
    % D_up_k(i_kave) = 1;
    i_kave2 = find(ne_param.K == ne_param.k_ave * 2);
    D_up_k_init = [1 / i_kave2 * ones(i_kave2, 1); zeros(ne_param.num_K - i_kave2, 1)];
    D_up_u_k = D_up_u * D_up_k_init.';
    D_up_u_k_next = D_up_u_k;
    T_down_u_k_up_un_kn = squeeze(dot2(reshape(D_up_u_k_next, [], 1), reshape(T_down_ui_ki_uj_kj_up_uin_kin, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
    D_up_u_k_next_next = squeeze(dot2(reshape(T_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(D_up_u_k_next, [], 1), 1, 1));
    D_up_u_k_next_next = D_up_u_k_next_next / sum(D_up_u_k_next_next(:));
    D_error = norm(D_up_u_k_next - D_up_u_k_next_next, inf);
    D_up_u_k_next = D_up_u_k_next_next;
    num_D_iter = 0;
    while D_error > ne_param.D_tol && num_D_iter < ne_param.D_max_iter
        T_down_u_k_up_un_kn = squeeze(dot2(reshape(D_up_u_k_next, [], 1), reshape(T_down_ui_ki_uj_kj_up_uin_kin, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
        D_up_u_k_next_next = squeeze(dot2(reshape(T_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(D_up_u_k_next, [], 1), 1, 1));
        D_up_u_k_next_next = D_up_u_k_next_next / sum(D_up_u_k_next_next(:));
        D_error = norm(D_up_u_k_next - D_up_u_k_next_next, inf);
        D_up_u_k_next = D_up_u_k_next_next;
        num_D_iter = num_D_iter + 1;
    end
    % Apply momentum
    D_up_u_k = ne_param.D_tau * D_up_u_k_next + (1 - ne_param.D_tau) * D_up_u_k;

    % Plot
    if ne_param.plot
        ne_D_plot_fg = 2;
        ne_D_plot_pos = [0, 0, default_width, default_height];
        ne_func.plot_ne_D(ne_D_plot_fg, ne_D_plot_pos, D_up_u_k, ne_param.U, ne_param.K, alpha);
    end

    %% Step 3
    % 3.1
    D_up_mj = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(pi_down_u_k_up_m, [], ne_param.num_K), 1, 1));
    q_down_ui_mi = dot2(D_up_mj, c_down_ui_mi_mj, 2, 3);
    q_down_ui_ki_mi = permute(outer(ones(ne_param.num_K, 1), q_down_ui_mi), [2 1 3]);
    T_down_ki_mi_up_uin_kin = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(T_down_ki_mi_uj_kj_up_uin_kin, ne_param.num_K, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
    T_down_ui_ki_mi_up_uin_kin = outer(ones(ne_param.num_U, 1), T_down_ki_mi_up_uin_kin);
    q_down_u_k = dot2(pi_down_u_k_up_m, q_down_ui_ki_mi, 3, 3);

    % 3.2
    V_down_u_k = zeros(ne_param.num_U, ne_param.num_K);
    V_down_u_k_next = q_down_u_k + alpha * dot2(reshape(T_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(V_down_u_k, [], 1), 3, 1);
    V_error = norm(V_down_u_k - V_down_u_k_next, inf);
    V_down_u_k = V_down_u_k_next;
    num_V_iter = 0;
    while V_error > ne_param.V_tol && num_V_iter < ne_param.V_max_iter
        V_down_u_k_next = q_down_u_k + alpha * dot2(reshape(T_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(V_down_u_k, [], 1), 3, 1);
        V_error = norm(V_down_u_k - V_down_u_k_next, inf);
        V_down_u_k = V_down_u_k_next;
        num_V_iter = num_V_iter + 1;
    end

    %% Step 4
    % 4.1
    V_down_u_k_m = q_down_ui_ki_mi + alpha * dot2(reshape(T_down_ui_ki_mi_up_uin_kin, ne_param.num_U, ne_param.num_K, ne_param.num_K, []), reshape(V_down_u_k, [], 1), 4, 1);

    % 4.2
    pi_i_down_ui_ki_up_mi = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
    for i_ui = 1 : ne_param.num_U
        for i_ki = 1 : ne_param.num_K
            [~, i_mi] = min(V_down_u_k_m(i_ui,i_ki,1:i_ki));
            pi_i_down_ui_ki_up_mi(i_ui,i_ki,i_mi) = 1;
        end
    end
    pi_i_diff = pi_down_u_k_up_m - pi_i_down_ui_ki_up_mi;
    pi_i_error = norm(reshape(pi_i_diff, [], 1), inf);

    % 4.3/4
    num_ne_pi_iter = 0;
    num_pi_iter = 0;
    % Display status
    fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_ne_pi_iter, num_pi_iter, pi_i_error);
    while pi_i_error > ne_param.policy_tol && num_pi_iter < ne_param.policy_max_iter
        % 4.4.1
        q_down_ui_ki = dot2(pi_i_down_ui_ki_up_mi, q_down_ui_ki_mi, 3, 3);
        T_down_ui_ki_up_uin_kin = squeeze(dot2(pi_i_down_ui_ki_up_mi, T_down_ui_ki_mi_up_uin_kin, 3, 3));

        % 4.4.2
        V_down_ui_ki = zeros(ne_param.num_U, ne_param.num_K);
        V_down_ui_ki_next = q_down_ui_ki + alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
        V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
        V_down_ui_ki = V_down_ui_ki_next;
        num_V_iter = 0;
        while V_error > ne_param.V_tol && num_V_iter < ne_param.V_max_iter
            V_down_ui_ki_next = q_down_ui_ki + alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
            V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
            V_down_ui_ki = V_down_ui_ki_next;
            num_V_iter = num_V_iter + 1;
        end

        % 4.4.3
        V_down_ui_ki_mi = q_down_ui_ki_mi + alpha * dot2(reshape(T_down_ui_ki_mi_up_uin_kin, ne_param.num_U, ne_param.num_K, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 4, 1);

        % 4.4.4
        pi_i_down_ui_ki_up_mi_next = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
        for i_ui = 1 : ne_param.num_U
            for i_ki = 1 : ne_param.num_K
                [~, i_mi] = min(V_down_ui_ki_mi(i_ui,i_ki,1:i_ki));
                pi_i_down_ui_ki_up_mi_next(i_ui,i_ki,i_mi) = 1;
            end
        end
        pi_i_diff = pi_i_down_ui_ki_up_mi - pi_i_down_ui_ki_up_mi_next;
        pi_i_error = norm(reshape(pi_i_diff, [], 1), inf);

        % 4.4.5
        pi_i_down_ui_ki_up_mi = pi_i_down_ui_ki_up_mi_next;
        num_pi_iter = num_pi_iter + 1;
        % Display status
        fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_ne_pi_iter, num_pi_iter, pi_i_error);
    end

    % Plot
    if ne_param.plot
        br_pi_plot_fg = 3;
        br_pi_plot_pos = [default_width, default_height, default_width, default_height];
        ne_func.plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, pi_i_down_ui_ki_up_mi, ne_param.U, ne_param.K, alpha);
        drawnow;
    end

    %% Step 5
    % Apply momentum
    pi_down_u_k_up_m_next = ne_param.policy_tau * pi_i_down_ui_ki_up_mi + (1 - ne_param.policy_tau) * pi_down_u_k_up_m;
    ne_pi_diff = pi_down_u_k_up_m - pi_down_u_k_up_m_next;
    ne_pi_error = rms(reshape(ne_pi_diff, [], 1));
    pi_down_u_k_up_m = pi_down_u_k_up_m_next;

    % Plot
    if ne_param.plot
        ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, pi_down_u_k_up_m, ne_param.U, ne_param.K, alpha);
    end
    % Display status and store history of policies
    fprintf('Iteration %d policy error %f\n', num_ne_pi_iter, ne_pi_error);
    pi_hist_end = zeros(1, ne_param.num_X);
    for i_ui = 1 : ne_param.num_U
        base_i_ui = (i_ui - 1) * ne_param.num_K;
        for i_ki = 1 : ne_param.num_K
            i_xi = base_i_ui + i_ki;
            [~, max_i] = max(pi_down_u_k_up_m(i_ui,i_ki,:));
            pi_hist_end(i_xi) = ne_param.K(max_i);
            if i_xi == 1
                fprintf('Iteration %d policy:\n%d', num_ne_pi_iter, pi_hist_end(i_xi));
            elseif mod(i_xi - 1, ne_param.num_K) == 0
                fprintf('\n%d', pi_hist_end(i_xi));
            else
                fprintf('->%d', pi_hist_end(i_xi));
            end
        end
    end
    pi_hist = pi_hist_end;
    fprintf('\n\n');
    ne_policy_error_hist = ne_pi_error;
    num_ne_pi_iter = num_ne_pi_iter + 1;
    while ne_pi_error > ne_param.policy_tol && num_ne_pi_iter < ne_param.ne_policy_max_iter
        %% Step 5.2
        % 5.2.1
        T_down_ki_mi_uj_kj_up_uin_kin = permute(squeeze(dot2(permute(phi_down_ki_mi_kj_mj_up_uin_kin, [1 2 5 6 3 4]), permute(pi_down_u_k_up_m, [2 3 1]), 6, 2)), [1 2 6 5 3 4]);
        T_down_ui_ki_uj_kj_up_uin_kin = permute(squeeze(dot2(permute(T_down_ki_mi_uj_kj_up_uin_kin, [3 4 5 6 1 2]), permute(pi_down_u_k_up_m, [2 3 1]), 6, 2)), [6 5 1 2 3 4]);

        % 5.2.2
        D_up_u_k_next = D_up_u * D_up_k_init.';
        T_down_u_k_up_un_kn = squeeze(dot2(reshape(D_up_u_k_next, [], 1), reshape(T_down_ui_ki_uj_kj_up_uin_kin, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
        D_up_u_k_next_next = squeeze(dot2(reshape(T_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(D_up_u_k_next, [], 1), 1, 1));
        D_up_u_k_next_next = D_up_u_k_next_next / sum(D_up_u_k_next_next(:));
        D_error = norm(D_up_u_k_next - D_up_u_k_next_next, inf);
        D_up_u_k_next = D_up_u_k_next_next;
        num_D_iter = 0;
        while D_error > ne_param.D_tol && num_D_iter < ne_param.D_max_iter
            T_down_u_k_up_un_kn = squeeze(dot2(reshape(D_up_u_k_next, [], 1), reshape(T_down_ui_ki_uj_kj_up_uin_kin, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
            D_up_u_k_next_next = squeeze(dot2(reshape(T_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(D_up_u_k_next, [], 1), 1, 1));
            D_up_u_k_next_next = D_up_u_k_next_next / sum(D_up_u_k_next_next(:));
            D_error = norm(D_up_u_k_next - D_up_u_k_next_next, inf);
            D_up_u_k_next = D_up_u_k_next_next;
            num_D_iter = num_D_iter + 1;
        end
        % Apply momentum
        D_up_u_k = ne_param.D_tau * D_up_u_k_next + (1 - ne_param.D_tau) * D_up_u_k;

        % Plot
        if ne_param.plot
            ne_func.plot_ne_D(ne_D_plot_fg, ne_D_plot_pos, D_up_u_k, ne_param.U, ne_param.K, alpha);
        end

        %% Step 5.3
        % 5.3.1
        D_up_mj = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(pi_down_u_k_up_m, [], ne_param.num_K), 1, 1));
        q_down_ui_mi = dot2(D_up_mj, c_down_ui_mi_mj, 2, 3);
        q_down_ui_ki_mi = permute(outer(ones(ne_param.num_K, 1), q_down_ui_mi), [2 1 3]);
        T_down_ki_mi_up_uin_kin = squeeze(dot2(reshape(D_up_u_k, [], 1), reshape(T_down_ki_mi_uj_kj_up_uin_kin, ne_param.num_K, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
        T_down_ui_ki_mi_up_uin_kin = outer(ones(ne_param.num_U, 1), T_down_ki_mi_up_uin_kin);
        q_down_u_k = dot2(pi_down_u_k_up_m, q_down_ui_ki_mi, 3, 3);

        % 5.3.2
        V_down_u_k = zeros(ne_param.num_U, ne_param.num_K);
        V_down_u_k_next = q_down_u_k + alpha * dot2(reshape(T_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(V_down_u_k, [], 1), 3, 1);
        V_error = norm(V_down_u_k - V_down_u_k_next, inf);
        V_down_u_k = V_down_u_k_next;
        num_V_iter = 0;
        while V_error > ne_param.V_tol && num_V_iter < ne_param.V_max_iter
            V_down_u_k_next = q_down_u_k + alpha * dot2(reshape(T_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(V_down_u_k, [], 1), 3, 1);
            V_error = norm(V_down_u_k - V_down_u_k_next, inf);
            V_down_u_k = V_down_u_k_next;
            num_V_iter = num_V_iter + 1;
        end

        %% Step 5.4
        % 5.4.1
        V_down_u_k_m = q_down_ui_ki_mi + alpha * dot2(reshape(T_down_ui_ki_mi_up_uin_kin, ne_param.num_U, ne_param.num_K, ne_param.num_K, []), reshape(V_down_u_k, [], 1), 4, 1);

        % 5.4.2
        pi_i_down_ui_ki_up_mi = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
        for i_ui = 1 : ne_param.num_U
            for i_ki = 1 : ne_param.num_K
                [~, i_mi] = min(V_down_u_k_m(i_ui,i_ki,1:i_ki));
                pi_i_down_ui_ki_up_mi(i_ui,i_ki,i_mi) = 1;
            end
        end
        pi_i_diff = pi_down_u_k_up_m - pi_i_down_ui_ki_up_mi;
        pi_i_error = norm(reshape(pi_i_diff, [], 1), inf);

        % 5.4.3/4
        num_pi_iter = 0;
        % Display status
        fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_ne_pi_iter, num_pi_iter, pi_i_error);
        while pi_i_error > ne_param.policy_tol && num_pi_iter < ne_param.policy_max_iter
            % 5.4.4.1
            q_down_ui_ki = dot2(pi_i_down_ui_ki_up_mi, q_down_ui_ki_mi, 3, 3);
            T_down_ui_ki_up_uin_kin = squeeze(dot2(pi_i_down_ui_ki_up_mi, T_down_ui_ki_mi_up_uin_kin, 3, 3));

            % 5.4.4.2
            V_down_ui_ki = zeros(ne_param.num_U, ne_param.num_K);
            V_down_ui_ki_next = q_down_ui_ki + alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
            V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
            V_down_ui_ki = V_down_ui_ki_next;
            num_V_iter = 0;
            while V_error > ne_param.V_tol && num_V_iter < ne_param.V_max_iter
                V_down_ui_ki_next = q_down_ui_ki + alpha * dot2(reshape(T_down_ui_ki_up_uin_kin, ne_param.num_U, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 3, 1);
                V_error = norm(V_down_ui_ki - V_down_ui_ki_next, inf);
                V_down_ui_ki = V_down_ui_ki_next;
                num_V_iter = num_V_iter + 1;
            end

            % 5.4.4.3
            V_down_ui_ki_mi = q_down_ui_ki_mi + alpha * dot2(reshape(T_down_ui_ki_mi_up_uin_kin, ne_param.num_U, ne_param.num_K, ne_param.num_K, []), reshape(V_down_ui_ki, [], 1), 4, 1);

            % 5.4.4.4
            pi_i_down_ui_ki_up_mi_next = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
            for i_ui = 1 : ne_param.num_U
                for i_ki = 1 : ne_param.num_K
                    [~, i_mi] = min(V_down_ui_ki_mi(i_ui,i_ki,1:i_ki));
                    pi_i_down_ui_ki_up_mi_next(i_ui,i_ki,i_mi) = 1;
                end
            end
            pi_i_diff = pi_i_down_ui_ki_up_mi - pi_i_down_ui_ki_up_mi_next;
            pi_i_error = norm(reshape(pi_i_diff, [], 1), inf);

            % 5.4.4.5
            pi_i_down_ui_ki_up_mi = pi_i_down_ui_ki_up_mi_next;
            num_pi_iter = num_pi_iter + 1;
            % Display status
            fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_ne_pi_iter, num_pi_iter, pi_i_error);
        end

        % Plot
        if ne_param.plot
            ne_func.plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, pi_i_down_ui_ki_up_mi, ne_param.U, ne_param.K, alpha);
            drawnow;
        end

        %% Step 5.5
        % Apply momentum
        pi_down_u_k_up_m_next = ne_param.policy_tau * pi_i_down_ui_ki_up_mi + (1 - ne_param.policy_tau) * pi_down_u_k_up_m;
        ne_pi_diff = pi_down_u_k_up_m - pi_down_u_k_up_m_next;
        ne_pi_error = rms(reshape(ne_pi_diff, [], 1));
        pi_down_u_k_up_m = pi_down_u_k_up_m_next;

        % Plot
        if ne_param.plot
            ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, pi_down_u_k_up_m, ne_param.U, ne_param.K, alpha);
        end

        % Display status and store history of policies
        fprintf('Iteration %d policy error %f\n', num_ne_pi_iter, ne_pi_error);
        for i_ui = 1 : ne_param.num_U
            base_i_ui = (i_ui - 1) * ne_param.num_K;
            for i_ki = 1 : ne_param.num_K
                i_xi = base_i_ui + i_ki;
                [~, max_i] = max(pi_down_u_k_up_m(i_ui,i_ki,:));
                pi_hist_end(i_xi) = ne_param.K(max_i);
                if i_xi == 1
                    fprintf('Iteration %d policy:\n%d', num_ne_pi_iter, pi_hist_end(i_xi));
                elseif mod(i_xi - 1, ne_param.num_K) == 0
                    fprintf('\n%d', pi_hist_end(i_xi));
                else
                    fprintf('->%d', pi_hist_end(i_xi));
                end
            end
        end
        % Detect a limit cycle
        limit_cycle = false;
        for pi_hist_i = 1 : size(pi_hist, 1)
            if isequal(pi_hist(pi_hist_i,:), pi_hist_end)
                % Limit cycle found
                limit_cycle = true;
                pi_limit_cycle = pi_hist(pi_hist_i:end,:);
                pi_limit_cycle_code = pi_limit_cycle * repmat((1 : ne_param.num_K).', ne_param.num_U, 1);
                break;
            end
        end
        pi_hist = [pi_hist; pi_hist_end];
        fprintf('\n\n');
        ne_policy_error_hist = [ne_policy_error_hist; ne_pi_error];
        num_ne_pi_iter = num_ne_pi_iter + 1;
        if ne_param.policy_tau == 1 && ne_param.D_tau == 1 && limit_cycle && size(pi_limit_cycle, 1) > 1
            fprintf('Limit cycle found!\n\n');
            break;
        end
    end
    
    % Plot remaining statistics
    if ne_param.plot
        % NE expected utility plot
        ne_V_plot_fg = 4;
        ne_V_plot_pos = [0, 0, default_width, default_height];
        ne_func.plot_ne_V(ne_V_plot_fg, ne_V_plot_pos, V_down_u_k, ne_param.U, ne_param.K, alpha)

        % NE expected utiliy per message plot
        ne_V_m_plot_fg = 5;
        ne_V_m_plot_pos = [default_width, 0, default_width, default_height];
        ne_func.plot_ne_V_m(ne_V_m_plot_fg, ne_V_m_plot_pos, parula, V_down_u_k_m, ne_param.U, ne_param.K, alpha)
        
        % NE state transitions plot
        ne_T_plot_fg = 6;
        ne_T_plot_pos = [0, 0, screenwidth, screenheight];
        ne_func.plot_ne_T(ne_T_plot_fg, ne_T_plot_pos, RedColormap, T_down_u_k_up_un_kn, ne_param.U, ne_param.K, alpha);
        
        % NE policy error plot
        ne_pi_error_plot_fg = 7;
        ne_pi_error_plot_pos = [default_width, 0, default_width, default_height];
        ne_func.plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_policy_error_hist, alpha);
    end
    
    % Store end results
    if ne_param.save
        save(['karma_nash_equilibrium/results/alpha_', num2str(alpha, '%.2f'), '.mat']);
    end
end

% If plotting is not active, plot everything at the end
if ~ne_param.plot
    % NE policy plot
    ne_pi_plot_fg = 1;
    ne_pi_plot_pos = [0, default_height, default_width, default_height];
    ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, pi_down_u_k_up_m, ne_param.U, ne_param.K, alpha);
    
    % NE stationary distribution plot
    ne_D_plot_fg = 2;
    ne_D_plot_pos = [0, 0, default_width, default_height];
    ne_func.plot_ne_D(ne_D_plot_fg, ne_D_plot_pos, D_up_u_k, ne_param.U, ne_param.K, alpha);
    
    % Agent i best response policy plot
    br_pi_plot_fg = 3;
    br_pi_plot_pos = [default_width, default_height, default_width, default_height];
    ne_func.plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, pi_i_down_ui_ki_up_mi, ne_param.U, ne_param.K, alpha);

    % NE expected utility plot
    ne_V_plot_fg = 4;
    ne_V_plot_pos = [0, 0, default_width, default_height];
    ne_func.plot_ne_V(ne_V_plot_fg, ne_V_plot_pos, V_down_u_k, ne_param.U, ne_param.K, alpha)

    % NE expected utiliy per message plot
    ne_V_m_plot_fg = 5;
    ne_V_m_plot_pos = [default_width, 0, default_width, default_height];
    ne_func.plot_ne_V_m(ne_V_m_plot_fg, ne_V_m_plot_pos, parula, V_down_u_k_m, ne_param.U, ne_param.K, alpha)

    % NE state transitions plot
    ne_T_plot_fg = 6;
    ne_T_plot_pos = [0, 0, screenwidth, screenheight];
    ne_func.plot_ne_T(ne_T_plot_fg, ne_T_plot_pos, RedColormap, T_down_u_k_up_un_kn, ne_param.U, ne_param.K, alpha);

    % NE policy error plot
    ne_pi_error_plot_fg = 7;
    ne_pi_error_plot_pos = [default_width, 0, default_width, default_height];
    ne_func.plot_ne_pi_error(ne_pi_error_plot_fg, ne_pi_error_plot_pos, ne_policy_error_hist, alpha);
end

%% Inform user when done
fprintf('DONE\n\n');