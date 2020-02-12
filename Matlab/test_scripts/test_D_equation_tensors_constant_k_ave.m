clc;
clear all;
close all;

%% Some parameters
Nu = 2;
K = 0 : 12;
Nk = length(K);
Nx = Nu * Nk;
k_ave = 10;
Nk_small = length(0 : k_ave - 1);
Nk_big = length(k_ave + 1 : K(end));
delta_constant = sum(0 : k_ave - 1) / Nk_small - sum(k_ave + 1 : K(end)) / Nk_big;
p_U = 0.5;
p_U = [p_U; 1 - p_U];

%% Create random T_down_xi_xj_up_xin that sum to 1 along xin dimension
% T_down_xi_xj_up_xin = rand(Nx, Nx, Nx);
% T_down_xi_xj_up_xin = T_down_xi_xj_up_xin ./ sum(T_down_xi_xj_up_xin, 3);

%% Load sample 
%load('T_down_ui_ki_uj_kj_up_uin_kin.mat');
load('karma_nash_equilibrium/results/k_max_12_k_ave_10/alpha_0.80.mat');
T_down_xi_xj_up_xin = zeros(Nx, Nx, Nx);
for i_ui = 1 : Nu
    base_i_ui = (i_ui - 1) * Nk;
    for i_ki = 1 : Nk
        i_xi = base_i_ui + i_ki;
        for i_uj = 1 : Nu
            base_i_uj = (i_uj - 1) * Nk;
            for i_kj = 1 : Nk
                i_xj = base_i_uj + i_kj;
                for i_uin = 1 : Nu
                    base_i_uin = (i_uin - 1) * Nk;
                    for i_kin = 1 :Nk
                        i_xin = base_i_uin + i_kin;
                        T_down_xi_xj_up_xin(i_xi,i_xj,i_xin) =...
                            sigma_down_u_k_uj_kj_up_un_kn(i_ui, i_ki, i_uj, i_kj, i_uin, i_kin);
                    end
                end
            end
        end
    end
end

% T_down_xi_uj_kj_up_uin_kin = reshape(T_down_ui_ki_uj_kj_up_uin_kin, [], Nu, Nk, Nu, Nk);
% T_down_xi_xj_up_uin_kin = reshape(T_down_xi_uj_kj_up_uin_kin, Nx, [], Nu, Nk);
% T_down_xi_xj_up_xin = reshape(T_down_xi_xj_up_uin_kin, Nx, Nx, []);

%% Iterative solution. Try many times to see sensitivity to initialiation
num_trials = 1000;
D_init = zeros(Nx, num_trials);
D = zeros(Nx, num_trials);
k_ave_init = zeros(1, num_trials);
k_ave_out = zeros(1, num_trials);
i_kave = find(K == k_ave);
i_trial = 1;

% Case D(k_ave) = 1
fprintf('Trial %d\n', i_trial);
D_init(i_kave,i_trial) = p_U(1);
D_init(Nk+i_kave,i_trial) = p_U(2);
D_init(:,i_trial) = D_init(:,i_trial) / sum(D_init(:,i_trial));
D_curr = D_init(:,i_trial);
k_ave_init(i_trial) = [K K] * D_curr;
D_next = zeros(Nx, 1);
for i_x = 1 : Nx
    D_next(i_x) = D_curr.' * T_down_xi_xj_up_xin(:,:,i_x) * D_curr;
end
D_next = D_next / sum(D_next);
while(norm(D_next - D_curr, inf) > 1e-6)
    D_curr = D_next;
    for i_x = 1 : Nx
        D_next(i_x) = D_curr.' * T_down_xi_xj_up_xin(:,:,i_x) * D_curr;
    end
    D_next = D_next / sum(D_next);
end
D(:,i_trial) = D_next;
k_ave_out(i_trial) = [K K] * D_next;
i_trial = i_trial + 1;

% Cases of pairing 2 values, one smaller than k_ave and one larger
for i_small = 1 : i_kave - 1
    for i_big = i_kave + 1 : Nk
        fprintf('Trial %d\n', i_trial);
        D_small_big = [K(i_small) K(i_big); 1 1] \ [k_ave; 1];
        D_init(i_small,i_trial) = p_U(1) * D_small_big(1);
        D_init(i_big,i_trial) = p_U(1) * D_small_big(2);
        D_init(Nk+i_small,i_trial) = p_U(2) * D_small_big(1);
        D_init(Nk+i_big,i_trial) = p_U(2) * D_small_big(2);
        D_init(:,i_trial) = D_init(:,i_trial) / sum(D_init(:,i_trial));
        D_curr = D_init(:,i_trial);
        k_ave_init(i_trial) = [K K] * D_curr;
        for i_x = 1 : Nx
            D_next(i_x) = D_curr.' * T_down_xi_xj_up_xin(:,:,i_x) * D_curr;
        end
        D_next = D_next / sum(D_next);
        while(norm(D_next - D_curr, inf) > 1e-6)
            D_curr = D_next;
            for i_x = 1 : Nx
                D_next(i_x) = D_curr.' * T_down_xi_xj_up_xin(:,:,i_x) * D_curr;
            end
            D_next = D_next / sum(D_next);
        end
        D(:,i_trial) = D_next;
        k_ave_out(i_trial) = [K K] * D_next;
        i_trial = i_trial + 1;
    end
end
i_trivial_trials = i_trial;

% Random cases
for i_trial = i_trivial_trials : num_trials
    fprintf('Trial %d\n', i_trial);
    D_k = rand(Nk, 1);
    D_k = D_k / sum(D_k);
    delta_k_ave = k_ave - K * D_k;
    delta_p = delta_k_ave / delta_constant;
    D_k(1:i_kave-1) = D_k(1:i_kave-1) + delta_p / Nk_small;
    D_k(i_kave+1:Nk) = D_k(i_kave+1:Nk) - delta_p / Nk_big;
    % Reject samples that lead to negative probabilities
    while min(D_k) < 0
        D_k = rand(Nk, 1);
        D_k = D_k / sum(D_k);
        delta_k_ave = k_ave - K * D_k;
        delta_p = delta_k_ave / delta_constant;
        D_k(1:i_kave-1) = D_k(1:i_kave-1) + delta_p / Nk_small;
        D_k(i_kave+1:Nk) = D_k(i_kave+1:Nk) - delta_p / Nk_big;
    end
    D_init(1:Nk,i_trial) = p_U(1) * D_k;
    D_init(Nk+1:end,i_trial) = p_U(2) * D_k;
    D_init(:,i_trial) = D_init(:,i_trial) / sum(D_init(:,i_trial));
    D_curr = D_init(:,i_trial);
    k_ave_init(i_trial) = [K K] * D_curr;
    for i_x = 1 : Nx
        D_next(i_x) = D_curr.' * T_down_xi_xj_up_xin(:,:,i_x) * D_curr;
    end
    D_next = D_next / sum(D_next);
    while(norm(D_next - D_curr, inf) > 1e-6)
        D_curr = D_next;
        for i_x = 1 : Nx
            D_next(i_x) = D_curr.' * T_down_xi_xj_up_xin(:,:,i_x) * D_curr;
        end
        D_next = D_next / sum(D_next);
    end
    D(:,i_trial) = D_next;
    k_ave_out(i_trial) = [K K] * D_next;
end
sensitivity = norm(max(D, [], 2) - min(D, [], 2), inf)
%D(:,10) = [];
% D(:,7) = [];
% sensitivity2 = norm(max(D, [], 2) - min(D, [], 2), inf)
sensitivity_k_ave_init = max(k_ave_init) - min(k_ave_init)
sensitivity_k_ave_out = max(k_ave_out) - min(k_ave_out)

fprintf('DONE ITERATIVE\n\n');

plot(D);

% %% QCQP solution
% D_opt = sdpvar(Nx, 1);
% F = ones(1, Nx) * D_opt == 1;   % D_opt entries sum to 1
% F = [F; eye(Nx) * D_opt >= zeros(Nx, 1)];  % D_opt entries are non-negative
% F = [F; [K K] * D_opt == k_ave];    % Average karma as specified
% for i_x = 1 : Nx
%     e_vec = zeros(Nx, 1);
%     e_vec(i_x) = 1;
%     F = [F; e_vec.' * D_opt == D_opt.' * T_down_xi_xj_up_xin(:,:,i_x) * D_opt];
% end
% assign(D_opt, D_init(:,1));
% options = sdpsettings('solver', 'fmincon', 'verbose', 1, 'usex0', 1);
% optimize(F, [], options);
% D_sol = value(D_opt);
% sensitivity_opt = norm(max([D, D_sol], [], 2) - min([D, D_sol], [], 2), inf);
% 
% fprintf('DONE QCQP\n\n');

% %% Some more testing for uniqueness
% T_down_ui_ki_uj_kj_up_kin = squeeze(sum(T_down_ui_ki_uj_kj_up_uin_kin, 5));
% T_down_ui_ki_kj_up_kin = squeeze(p_U(1) * T_down_ui_ki_uj_kj_up_kin(:,:,1,:,:) + p_U(2) * T_down_ui_ki_uj_kj_up_kin(:,:,2,:,:));
% T_down_ki_kj_up_kin = squeeze(p_U(1) * T_down_ui_ki_kj_up_kin(1,:,:,:) + p_U(2) * T_down_ui_ki_kj_up_kin(2,:,:,:));
% K_T = zeros(Nk);
% for i_k = 1 : Nk
%     K_T = K_T + K(i_k) * T_down_ki_kj_up_kin(:,:,i_k);
% end
% D_up_k = D_curr(1:Nk) + D_curr(Nk+1:end);
% D_up_k.' * K_T * D_up_k;
% 
% fprintf('DONE\n\n');