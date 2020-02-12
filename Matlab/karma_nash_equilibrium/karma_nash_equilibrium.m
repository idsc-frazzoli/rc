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

%% Loop over alphas
for i_alpha = 1 : ne_param.num_alpha
    alpha = ne_param.alpha(i_alpha);
    fprintf('%%%%ALPHA = %f%%%%\n\n', alpha);

    %% Step 1: NE policy guess
    ne_pi_down_u_k_up_m = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_M);
    for i_k = 1 : ne_param.num_K
        % Initial policy for nonurgent is to always bid 0
        ne_pi_down_u_k_up_m(1,i_k,1) = 1;
        % Initial policy for urgent is uniform distribution over messages
        i_adm_m = find(ne_param.M <= ne_param.K(i_k));
        p = 1 / length(i_adm_m);
        ne_pi_down_u_k_up_m(2,i_k,i_adm_m) = p;
%         % Initial policy for urgent is to always bid 1
%         if i_k > 1
%             ne_pi_down_u_k_up_m(2,i_k,2) = 1;
%         else
%             ne_pi_down_u_k_up_m(2,i_k,1) = 1;
%         end
    end

    % Plot
    if ne_param.plot
        ne_pi_plot_fg = 1;
        ne_pi_plot_pos = [0, default_height, default_width, default_height];
        ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);
    end

    %% Step 2
    % 2.1
    lambda_down_u_k_m_uj_kj_up_un_kn = permute(squeeze(dot2(ne_pi_down_u_k_up_m, permute(psi_down_u_k_m_kj_mj_up_un_kn, [4 5 1 2 3 6 7]), 3, 2)), [3 4 5 1 2 6 7]);
    sigma_down_u_k_uj_kj_up_un_kn = squeeze(dot2(ne_pi_down_u_k_up_m, lambda_down_u_k_m_uj_kj_up_un_kn, 3, 3));

    % 2.2
    D_up_u_init = ne_func.stat_dist(ne_param.mu_down_u_up_un);
    i_kave = find(ne_param.K == ne_param.k_ave);
    if ne_param.k_ave * 2 <= ne_param.k_max
        i_kave2 = find(ne_param.K == ne_param.k_ave * 2);
        D_up_k_init = [1 / i_kave2 * ones(i_kave2, 1); zeros(ne_param.num_K - i_kave2, 1)];
    elseif ne_param.k_ave >= ne_param.k_max
        D_up_k_init = zeros(ne_param.num_K, 1);
        D_up_k_init(end) = 1;
    else
        D_up_k_init = 1 / ne_param.num_K * ones(ne_param.num_K, 1);
        K_small = 0 : ne_param.k_ave - 1;
        K_big = ne_param.k_ave + 1 : ne_param.k_max;
        num_K_small = length(K_small);
        num_K_big = length(K_big);
        delta_constant = sum(K_small) / num_K_small - sum(K_big) / num_K_big;
        delta_k_ave = ne_param.k_ave - ne_param.K.' * D_up_k_init;
        delta_p = delta_k_ave / delta_constant;
        D_up_k_init(1:i_kave-1) = D_up_k_init(1:i_kave-1) + delta_p / num_K_small;
        D_up_k_init(i_kave+1:end) = D_up_k_init(i_kave+1:end) - delta_p / num_K_big;
    end
    D_up_u_k_init = outer(D_up_u_init, D_up_k_init);
    ne_d_up_u_k = D_up_u_k_init;
    ne_d_up_u_k_next = ne_d_up_u_k;
    
    % 2.3
    t_down_u_k_up_un_kn = squeeze(dot2(reshape(ne_d_up_u_k_next, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
    ne_d_up_u_k_next_next = squeeze(dot2(reshape(t_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(ne_d_up_u_k_next, [], 1), 1, 1));
    ne_d_up_u_k_next_next = ne_d_up_u_k_next_next / sum(ne_d_up_u_k_next_next(:));
    ne_d_error = norm(ne_d_up_u_k_next - ne_d_up_u_k_next_next, inf);
    ne_d_up_u_k_next = ne_d_up_u_k_next_next;
    num_d_iter = 0;
    while ne_d_error > ne_param.d_tol && num_d_iter < ne_param.d_max_iter
        t_down_u_k_up_un_kn = squeeze(dot2(reshape(ne_d_up_u_k_next, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
        ne_d_up_u_k_next_next = squeeze(dot2(reshape(t_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(ne_d_up_u_k_next, [], 1), 1, 1));
        ne_d_up_u_k_next_next = ne_d_up_u_k_next_next / sum(ne_d_up_u_k_next_next(:));
        ne_d_error = norm(ne_d_up_u_k_next - ne_d_up_u_k_next_next, inf);
        ne_d_up_u_k_next = ne_d_up_u_k_next_next;
        num_d_iter = num_d_iter + 1;
    end
    % Apply momentum
    ne_d_up_u_k = ne_param.d_mom * ne_d_up_u_k_next + (1 - ne_param.d_mom) * ne_d_up_u_k;

    % Plot
    if ne_param.plot
        ne_d_plot_fg = 2;
        ne_d_plot_pos = [0, 0, default_width, default_height];
        ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
    end

    %% Step 3
    % 3.1
    iota_up_mj = squeeze(dot2(reshape(ne_d_up_u_k, [], 1), reshape(ne_pi_down_u_k_up_m, [], ne_param.num_M), 1, 1));
    xi_down_u_m = dot2(iota_up_mj, c_down_u_m_mj, 2, 3);

    % 3.2
    ne_q_down_u_k = squeeze(dot2(xi_down_u_m, permute(ne_pi_down_u_k_up_m, [1 3 2]), 2, 2));
    ne_t_down_u_k_up_un_kn = squeeze(dot2(reshape(ne_d_up_u_k, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
    
    % 3.3
    ne_v_down_u_k = zeros(ne_param.num_U, ne_param.num_K);
    ne_v_down_u_k_next = ne_q_down_u_k + alpha * dot2(reshape(ne_t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 3, 1);
    v_error = norm(ne_v_down_u_k - ne_v_down_u_k_next, inf);
    ne_v_down_u_k = ne_v_down_u_k_next;
    num_v_iter = 0;
    while v_error > ne_param.v_tol && num_v_iter < ne_param.v_max_iter
        ne_v_down_u_k_next = ne_q_down_u_k + alpha * dot2(reshape(ne_t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 3, 1);
        v_error = norm(ne_v_down_u_k - ne_v_down_u_k_next, inf);
        ne_v_down_u_k = ne_v_down_u_k_next;
        num_v_iter = num_v_iter + 1;
    end

    %% Step 4
    % 4.1
    chi_down_u_k_m_up_un_kn = squeeze(dot2(reshape(ne_d_up_u_k, [], 1), reshape(lambda_down_u_k_m_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, ne_param.num_M, [], ne_param.num_U, ne_param.num_K), 1, 4));
    omega_down_u_k_m = dot2(reshape(chi_down_u_k_m_up_un_kn, ne_param.num_U, ne_param.num_K, ne_param.num_M, []), reshape(ne_v_down_u_k, [], 1), 4, 1);
    eta_down_u_k_m = permute(outer(ones(ne_param.num_K, 1), xi_down_u_m), [2 1 3]);

    % 4.2
    ne_rho_down_u_k_m = eta_down_u_k_m + alpha * omega_down_u_k_m;

    % 4.3
    br_pi_down_u_k_up_m = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_M);
    for i_u = 1 : ne_param.num_U
        for i_k = 1 : ne_param.num_K
            i_adm_m = find(ne_param.M <= ne_param.K(i_k));
            [min_rho, i_min_m] = min(ne_rho_down_u_k_m(i_u,i_k,i_adm_m));
            if abs(min_rho - ne_v_down_u_k(i_u,i_k)) <= ne_param.br_v_tol
                br_pi_down_u_k_up_m(i_u,i_k,:) = ne_pi_down_u_k_up_m(i_u,i_k,:);
            else
                br_pi_down_u_k_up_m(i_u,i_k,i_min_m) = 1;
            end
        end
    end
    br_pi_diff = ne_pi_down_u_k_up_m - br_pi_down_u_k_up_m;
    br_pi_error = norm(reshape(br_pi_diff, [], 1), inf);

    % 4.4/5
    num_ne_pi_iter = 0;
    num_br_pi_iter = 0;
    % Display status
    fprintf('Iteration %d best response policy iteration %d best response policy error %f\n', num_ne_pi_iter, num_br_pi_iter, br_pi_error);
    while br_pi_error > ne_param.br_pi_tol && num_br_pi_iter < ne_param.br_pi_max_iter
        % 4.5.1
        q_down_u_k = dot2(br_pi_down_u_k_up_m, eta_down_u_k_m, 3, 3);
        t_down_u_k_up_un_kn = squeeze(dot2(br_pi_down_u_k_up_m, chi_down_u_k_m_up_un_kn, 3, 3));

        % 4.5.2
        v_down_u_k = zeros(ne_param.num_U, ne_param.num_K);
        v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
        v_error = norm(v_down_u_k - v_down_u_k_next, inf);
        v_down_u_k = v_down_u_k_next;
        num_v_iter = 0;
        while v_error > ne_param.v_tol && num_v_iter < ne_param.v_max_iter
            v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
            v_error = norm(v_down_u_k - v_down_u_k_next, inf);
            v_down_u_k = v_down_u_k_next;
            num_v_iter = num_v_iter + 1;
        end

        % 4.5.3
        omega_down_u_k_m = dot2(reshape(chi_down_u_k_m_up_un_kn, ne_param.num_U, ne_param.num_K, ne_param.num_M, []), reshape(ne_v_down_u_k, [], 1), 4, 1);
        rho_down_u_k_m = eta_down_u_k_m + alpha * omega_down_u_k_m;

        % 4.5.4
        br_pi_down_u_k_up_m_next = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_M);
        for i_u = 1 : ne_param.num_U
            for i_k = 1 : ne_param.num_K
                i_adm_m = find(ne_param.M <= ne_param.K(i_k));
                [min_rho, i_min_m] = min(rho_down_u_k_m(i_u,i_k,i_adm_m));
                if abs(min_rho - ne_v_down_u_k(i_u,i_k)) <= ne_param.br_v_tol
                    br_pi_down_u_k_up_m_next(i_u,i_k,:) = br_pi_down_u_k_up_m(i_u,i_k,:);
                else
                    br_pi_down_u_k_up_m_next(i_u,i_k,i_min_m) = 1;
                end
            end
        end
        br_pi_diff = br_pi_down_u_k_up_m - br_pi_down_u_k_up_m_next;
        br_pi_error = norm(reshape(br_pi_diff, [], 1), inf);

        % 4.4.5
        br_pi_down_u_k_up_m = br_pi_down_u_k_up_m_next;
        num_br_pi_iter = num_br_pi_iter + 1;
        % Display status
        fprintf('Iteration %d best response policy iteration %d best response policy error %f\n', num_ne_pi_iter, num_br_pi_iter, br_pi_error);
    end

    % Plot
    if ne_param.plot
        br_pi_plot_fg = 3;
        br_pi_plot_pos = [default_width, default_height, default_width, default_height];
        ne_func.plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, br_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);
        drawnow;
    end

    %% Step 5
    % Apply momentum
    ne_pi_down_u_k_up_m_next = ne_param.ne_pi_mom * br_pi_down_u_k_up_m + (1 - ne_param.ne_pi_mom) * ne_pi_down_u_k_up_m;
    ne_pi_diff = ne_pi_down_u_k_up_m - ne_pi_down_u_k_up_m_next;
    ne_pi_error = rms(reshape(ne_pi_diff, [], 1));
    ne_pi_down_u_k_up_m = ne_pi_down_u_k_up_m_next;

    % Plot
    if ne_param.plot
        ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);
    end
    % Display status and store history of policies
    fprintf('Iteration %d policy error %f\n', num_ne_pi_iter, ne_pi_error);
    ne_pi_hist_end = zeros(1, ne_param.num_X);
    for i_u = 1 : ne_param.num_U
        base_i_u = (i_u - 1) * ne_param.num_K;
        for i_k = 1 : ne_param.num_K
            i_x = base_i_u + i_k;
            [~, i_max_m] = max(ne_pi_down_u_k_up_m(i_u,i_k,:));
            ne_pi_hist_end(i_x) = ne_param.M(i_max_m);
            if i_x == 1
                fprintf('Iteration %d policy:\n%.2f', num_ne_pi_iter, ne_pi_hist_end(i_x));
            elseif mod(i_x - 1, ne_param.num_K) == 0
                fprintf('\n%.2f', ne_pi_hist_end(i_x));
            else
                fprintf('->%.2f', ne_pi_hist_end(i_x));
            end
        end
    end
    ne_pi_hist = ne_pi_hist_end;
    fprintf('\n\n');
    ne_pi_error_hist = ne_pi_error;
    num_ne_pi_iter = num_ne_pi_iter + 1;
    while ne_pi_error > ne_param.ne_pi_tol && num_ne_pi_iter < ne_param.ne_pi_max_iter
        %% Step 5.2
        % 5.2.1
        lambda_down_u_k_m_uj_kj_up_un_kn = permute(squeeze(dot2(ne_pi_down_u_k_up_m, permute(psi_down_u_k_m_kj_mj_up_un_kn, [4 5 1 2 3 6 7]), 3, 2)), [3 4 5 1 2 6 7]);
        sigma_down_u_k_uj_kj_up_un_kn = squeeze(dot2(ne_pi_down_u_k_up_m, lambda_down_u_k_m_uj_kj_up_un_kn, 3, 3));

        % 5.2.2
        ne_d_up_u_k_next = D_up_u_k_init;
        
        % 5.2.3
        t_down_u_k_up_un_kn = squeeze(dot2(reshape(ne_d_up_u_k_next, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
        ne_d_up_u_k_next_next = squeeze(dot2(reshape(t_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(ne_d_up_u_k_next, [], 1), 1, 1));
        ne_d_up_u_k_next_next = ne_d_up_u_k_next_next / sum(ne_d_up_u_k_next_next(:));
        ne_d_error = norm(ne_d_up_u_k_next - ne_d_up_u_k_next_next, inf);
        ne_d_up_u_k_next = ne_d_up_u_k_next_next;
        num_d_iter = 0;
        while ne_d_error > ne_param.d_tol && num_d_iter < ne_param.d_max_iter
            t_down_u_k_up_un_kn = squeeze(dot2(reshape(ne_d_up_u_k_next, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
            ne_d_up_u_k_next_next = squeeze(dot2(reshape(t_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(ne_d_up_u_k_next, [], 1), 1, 1));
            ne_d_up_u_k_next_next = ne_d_up_u_k_next_next / sum(ne_d_up_u_k_next_next(:));
            ne_d_error = norm(ne_d_up_u_k_next - ne_d_up_u_k_next_next, inf);
            ne_d_up_u_k_next = ne_d_up_u_k_next_next;
            num_d_iter = num_d_iter + 1;
        end
        % Apply momentum
        ne_d_up_u_k = ne_param.d_mom * ne_d_up_u_k_next + (1 - ne_param.d_mom) * ne_d_up_u_k;

        % Plot
        if ne_param.plot
            ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
        end

        %% Step 5.3
        % 5.3.1
        iota_up_mj = squeeze(dot2(reshape(ne_d_up_u_k, [], 1), reshape(ne_pi_down_u_k_up_m, [], ne_param.num_M), 1, 1));
        xi_down_u_m = dot2(iota_up_mj, c_down_u_m_mj, 2, 3);

        % 5.3.2
        ne_q_down_u_k = squeeze(dot2(xi_down_u_m, permute(ne_pi_down_u_k_up_m, [1 3 2]), 2, 2));
        ne_t_down_u_k_up_un_kn = squeeze(dot2(reshape(ne_d_up_u_k, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
        
        % 5.3.3
        ne_v_down_u_k = zeros(ne_param.num_U, ne_param.num_K);
        ne_v_down_u_k_next = ne_q_down_u_k + alpha * dot2(reshape(ne_t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 3, 1);
        v_error = norm(ne_v_down_u_k - ne_v_down_u_k_next, inf);
        ne_v_down_u_k = ne_v_down_u_k_next;
        num_v_iter = 0;
        while v_error > ne_param.v_tol && num_v_iter < ne_param.v_max_iter
            ne_v_down_u_k_next = ne_q_down_u_k + alpha * dot2(reshape(ne_t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(ne_v_down_u_k, [], 1), 3, 1);
            v_error = norm(ne_v_down_u_k - ne_v_down_u_k_next, inf);
            ne_v_down_u_k = ne_v_down_u_k_next;
            num_v_iter = num_v_iter + 1;
        end

        %% Step 5.4
        % 5.4.1
        chi_down_u_k_m_up_un_kn = squeeze(dot2(reshape(ne_d_up_u_k, [], 1), reshape(lambda_down_u_k_m_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, ne_param.num_M, [], ne_param.num_U, ne_param.num_K), 1, 4));
        omega_down_u_k_m = dot2(reshape(chi_down_u_k_m_up_un_kn, ne_param.num_U, ne_param.num_K, ne_param.num_M, []), reshape(ne_v_down_u_k, [], 1), 4, 1);
        eta_down_u_k_m = permute(outer(ones(ne_param.num_K, 1), xi_down_u_m), [2 1 3]);
        
        % 5.4.2
        ne_rho_down_u_k_m = eta_down_u_k_m + alpha * omega_down_u_k_m;

        % 5.4.3
        br_pi_down_u_k_up_m = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_M);
        for i_u = 1 : ne_param.num_U
            for i_k = 1 : ne_param.num_K
                i_m_adm = find(ne_param.M <= ne_param.K(i_k));
                [min_rho, i_min_m] = min(ne_rho_down_u_k_m(i_u,i_k,i_m_adm));
                if abs(min_rho - ne_v_down_u_k(i_u,i_k)) <= ne_param.br_v_tol
                    br_pi_down_u_k_up_m(i_u,i_k,:) = ne_pi_down_u_k_up_m(i_u,i_k,:);
                else
                    br_pi_down_u_k_up_m(i_u,i_k,i_min_m) = 1;
                end
            end
        end
        br_pi_diff = ne_pi_down_u_k_up_m - br_pi_down_u_k_up_m;
        br_pi_error = norm(reshape(br_pi_diff, [], 1), inf);

        % 5.4.4/5
        num_br_pi_iter = 0;
        % Display status
        fprintf('Iteration %d best response policy iteration %d best response policy error %f\n', num_ne_pi_iter, num_br_pi_iter, br_pi_error);
        while br_pi_error > ne_param.br_pi_tol && num_br_pi_iter < ne_param.br_pi_max_iter
            % 5.4.5.1
            q_down_u_k = dot2(br_pi_down_u_k_up_m, eta_down_u_k_m, 3, 3);
            t_down_u_k_up_un_kn = squeeze(dot2(br_pi_down_u_k_up_m, chi_down_u_k_m_up_un_kn, 3, 3));

            % 5.4.5.2
            v_down_u_k = zeros(ne_param.num_U, ne_param.num_K);
            v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
            v_error = norm(v_down_u_k - v_down_u_k_next, inf);
            v_down_u_k = v_down_u_k_next;
            num_v_iter = 0;
            while v_error > ne_param.v_tol && num_v_iter < ne_param.v_max_iter
                v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
                v_error = norm(v_down_u_k - v_down_u_k_next, inf);
                v_down_u_k = v_down_u_k_next;
                num_v_iter = num_v_iter + 1;
            end

            % 5.4.5.3
            omega_down_u_k_m = dot2(reshape(chi_down_u_k_m_up_un_kn, ne_param.num_U, ne_param.num_K, ne_param.num_M, []), reshape(ne_v_down_u_k, [], 1), 4, 1);
            rho_down_u_k_m = eta_down_u_k_m + alpha * omega_down_u_k_m;

            % 5.4.5.4
            br_pi_down_u_k_up_m_next = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_M);
            for i_u = 1 : ne_param.num_U
                for i_k = 1 : ne_param.num_K
                    i_m_adm = find(ne_param.M <= ne_param.K(i_k));
                    [min_rho, i_min_m] = min(rho_down_u_k_m(i_u,i_k,i_m_adm));
                    if abs(min_rho - ne_v_down_u_k(i_u,i_k)) <= ne_param.br_v_tol
                        br_pi_down_u_k_up_m_next(i_u,i_k,:) = br_pi_down_u_k_up_m(i_u,i_k,:);
                    else
                        br_pi_down_u_k_up_m_next(i_u,i_k,i_min_m) = 1;
                    end
                end
            end
            br_pi_diff = br_pi_down_u_k_up_m - br_pi_down_u_k_up_m_next;
            br_pi_error = norm(reshape(br_pi_diff, [], 1), inf);

            % 5.4.4.5
            br_pi_down_u_k_up_m = br_pi_down_u_k_up_m_next;
            num_br_pi_iter = num_br_pi_iter + 1;
            % Display status
            fprintf('Iteration %d best response policy iteration %d best response policy error %f\n', num_ne_pi_iter, num_br_pi_iter, br_pi_error);
        end

        % Plot
        if ne_param.plot
            ne_func.plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, br_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);
            drawnow;
        end

        %% Step 5.5
        % Apply momentum
        ne_pi_down_u_k_up_m_next = ne_param.ne_pi_mom * br_pi_down_u_k_up_m + (1 - ne_param.ne_pi_mom) * ne_pi_down_u_k_up_m;
        ne_pi_diff = ne_pi_down_u_k_up_m - ne_pi_down_u_k_up_m_next;
        ne_pi_error = rms(reshape(ne_pi_diff, [], 1));
        ne_pi_down_u_k_up_m = ne_pi_down_u_k_up_m_next;

        % Plot
        if ne_param.plot
            ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);
        end

        % Display status and store history of policies
        fprintf('Iteration %d policy error %f\n', num_ne_pi_iter, ne_pi_error);
        for i_u = 1 : ne_param.num_U
            base_i_u = (i_u - 1) * ne_param.num_K;
            for i_k = 1 : ne_param.num_K
                i_x = base_i_u + i_k;
                [~, i_max_m] = max(ne_pi_down_u_k_up_m(i_u,i_k,:));
                ne_pi_hist_end(i_x) = ne_param.M(i_max_m);
                if i_x == 1
                    fprintf('Iteration %d policy:\n%.2f', num_ne_pi_iter, ne_pi_hist_end(i_x));
                elseif mod(i_x - 1, ne_param.num_K) == 0
                    fprintf('\n%.2f', ne_pi_hist_end(i_x));
                else
                    fprintf('->%.2f', ne_pi_hist_end(i_x));
                end
            end
        end
        % Detect a limit cycle (only if momentum is turned off)
        limit_cycle = false;
        if ne_param.ne_pi_mom == 1 && ne_param.d_mom == 1
            for pi_hist_i = 1 : size(ne_pi_hist, 1)
                if isequal(ne_pi_hist(pi_hist_i,:), ne_pi_hist_end)
                    % Limit cycle found
                    limit_cycle = true;
                    ne_pi_limit_cycle = ne_pi_hist(pi_hist_i:end,:);
                    ne_pi_limit_cycle_code = ne_pi_limit_cycle * repmat((1 : ne_param.num_K).', ne_param.num_U, 1);
                    break;
                end
            end
        end
        ne_pi_hist = [ne_pi_hist; ne_pi_hist_end];
        fprintf('\n\n');
        ne_pi_error_hist = [ne_pi_error_hist; ne_pi_error];
        num_ne_pi_iter = num_ne_pi_iter + 1;
        if  limit_cycle && size(ne_pi_limit_cycle, 1) > 1
            fprintf('Limit cycle found!\n\n');
            break;
        end
    end
    
    % Plot remaining statistics
    if ne_param.plot
        % NE expected utility plot
        ne_v_plot_fg = 4;
        ne_v_plot_pos = [0, 0, default_width, default_height];
        ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_v_down_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);

        % NE expected utiliy per message plot
        ne_rho_plot_fg = 5;
        ne_rho_plot_pos = [default_width, 0, default_width, default_height];
        ne_func.plot_ne_rho(ne_rho_plot_fg, ne_rho_plot_pos, parula, ne_rho_down_u_k_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);
        
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
        save(['karma_nash_equilibrium/results/alpha_', num2str(alpha, '%.2f'), '.mat']);
    end
end

%% If plotting is not active, plot everything at the end
if ~ne_param.plot
    % NE policy plot
    ne_pi_plot_fg = 1;
    ne_pi_plot_pos = [0, default_height, default_width, default_height];
    ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);
    
    % NE stationary distribution plot
    ne_d_plot_fg = 2;
    ne_d_plot_pos = [0, 0, default_width, default_height];
    ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);
    
    % Agent i best response policy plot
    br_pi_plot_fg = 3;
    br_pi_plot_pos = [default_width, default_height, default_width, default_height];
    ne_func.plot_br_pi(br_pi_plot_fg, br_pi_plot_pos, RedColormap, br_pi_down_u_k_up_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);

    % NE expected utility plot
    ne_v_plot_fg = 4;
    ne_v_plot_pos = [0, 0, default_width, default_height];
    ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_v_down_u_k, ne_param.U, ne_param.K, ne_param.k_ave, alpha);

    % NE expected utiliy per message plot
    ne_rho_plot_fg = 5;
    ne_rho_plot_pos = [default_width, 0, default_width, default_height];
    ne_func.plot_ne_rho(ne_rho_plot_fg, ne_rho_plot_pos, parula, ne_rho_down_u_k_m, ne_param.U, ne_param.K, ne_param.M, ne_param.k_ave, alpha);

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