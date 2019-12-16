clc;
clear all;
close all;

%% Some parameters
Nu = 2;
Nk = 13;
Nx = Nu * Nk;

%% Create random T_down_xi_xj_up_xin that sum to 1 along xin dimension
% T_down_xi_xj_up_xin = rand(Nx, Nx, Nx);
% T_down_xi_xj_up_xin = 1 / Nx * ones(Nx, Nx, Nx);
T_down_xi_xj_up_xin = zeros(Nx, Nx, Nx);
T_down_xi_xj_up_xin(:,:,1) = 1;
T_down_xi_xj_up_xin = T_down_xi_xj_up_xin ./ sum(T_down_xi_xj_up_xin, 3);

%% Load sample 
% load('T_down_ui_ki_uj_kj_up_uin_kin.mat');
% T_down_xi_xj_up_xin = zeros(Nx, Nx, Nx);
% for i_ui = 1 : Nu
%     base_i_ui = (i_ui - 1) * Nk;
%     for i_ki = 1 : Nk
%         i_xi = base_i_ui + i_ki;
%         for i_uj = 1 : Nu
%             base_i_uj = (i_uj - 1) * Nk;
%             for i_kj = 1 : Nk
%                 i_xj = base_i_uj + i_kj;
%                 for i_uin = 1 : Nu
%                     base_i_uin = (i_uin - 1) * Nk;
%                     for i_kin = 1 :Nk
%                         i_xin = base_i_uin + i_kin;
%                         T_down_xi_xj_up_xin(i_xi,i_xj,i_xin) =...
%                             T_down_ui_ki_uj_kj_up_uin_kin(i_ui, i_ki, i_uj, i_kj, i_uin, i_kin);
%                     end
%                 end
%             end
%         end
%     end
% end

% T_down_xi_uj_kj_up_uin_kin = reshape(T_down_ui_ki_uj_kj_up_uin_kin, [], Nu, Nk, Nu, Nk);
% T_down_xi_xj_up_uin_kin = reshape(T_down_xi_uj_kj_up_uin_kin, Nx, [], Nu, Nk);
% T_down_xi_xj_up_xin = reshape(T_down_xi_xj_up_uin_kin, Nx, Nx, []);

%% Iterative solution. Try many times to see sensitivity to initialiation
num_trials = 1000;
D_init = zeros(Nx, num_trials);
D = zeros(Nx, num_trials);
for i = 1 : num_trials
    fprintf('Trial %d\n', i);
    if i <= Nx
        D_init(i,i) = 1;
    elseif i == Nx + 1
        D_init(:,i) = 1 / Nx * ones(Nx, 1);
    elseif i == Nx + 2
        D_init(1:Nk,i) = 0.5 * 1 / Nk * ones(Nk, 1);
        D_init(Nk+1:end,i) = 0.5 * 1 / Nk * ones(Nk, 1);
    else
        D_init(:,i) = rand(Nx, 1);
    end
    D_init(:,i) = D_init(:,i) / sum(D_init(:,i));
    D_curr = D_init(:,i);
    D_next = zeros(Nx, 1);
    for i_x = 1 : Nx
        D_next(i_x) = D_curr.' * T_down_xi_xj_up_xin(:,:,i_x) * D_curr;
    end
    D_next = D_next / sum(D_next);
    while(norm(D_next - D_curr, inf) > 1e-4)
        D_curr = D_next;
        for i_x = 1 : Nx
            D_next(i_x) = D_curr.' * T_down_xi_xj_up_xin(:,:,i_x) * D_curr;
        end
        D_next = D_next / sum(D_next);
    end
    D(:,i) = D_next;
end
sensitivity = norm(max(D, [], 2) - min(D, [], 2), inf);
sensitivity2 = norm(max(D(:,Nx+1:end), [], 2) - min(D(:,Nx+1:end), [], 2), inf);

fprintf('DONE ITERATIVE\n\n');

% %% QCQP solution
% D_opt = sdpvar(Nx, 1);
% F = ones(1, Nx) * D_opt == 1;   % D_opt entries sum to 1
% F = [F; eye(Nx) * D_opt >= zeros(Nx, 1)];  % D_opt entries are non-negative
% for i_x = 1 : Nx
%     e_vec = zeros(Nx, 1);
%     e_vec(i_x) = 1;
%     F = [F; e_vec.' * D_opt == D_opt.' * T_down_xi_xj_up_xin(:,:,i_x) * D_opt];
% end
% options = sdpsettings('solver', 'fmincon', 'verbose', 1);
% optimize(F, [], options);
% D_sol = value(D_opt);
% 
% sensitivity_opt = norm(max([D, D_sol], [], 2) - min([D, D_sol], [], 2), inf);
% sensitivity_opt2 = norm(max([D(:,Nx+1:end), D_sol], [], 2) - min([D(:,Nx+1:end), D_sol], [], 2), inf);
% 
% fprintf('DONE QCQP\n\n');

%% Check uniqueness condition
sum_T_t_T = zeros(Nx);
max_eig_T_t_T = zeros(Nx, 1);
for i_x = 1 : Nx
    T_t_T = T_down_xi_xj_up_xin(:,:,i_x).' * T_down_xi_xj_up_xin(:,:,i_x);
    sum_T_t_T = sum_T_t_T + T_t_T;
    max_eig_T_t_T(i_x) = max(eig(T_t_T));
end

eig_sum_T_t_T = eig(sum_T_t_T);
max_eig_sum_T_t_T = max(eig_sum_T_t_T);

%% Check other uniqueness condition
sum_cond = 0;
max_cond = zeros(Nx, 1);
for i_x = 1 : Nx
    T_sym = T_down_xi_xj_up_xin(:,:,i_x) + T_down_xi_xj_up_xin(:,:,i_x).' / 2;
    max_eig_T_sym = max(eig(T_sym));
    sum_cond = sum_cond + max_eig_T_sym^2;
    max_cond(i_x) = max_eig_T_sym;
end

cond_1 = 2 * sqrt(sum_cond) / Nx
cond_2 = 2 * max(max_cond)

fprintf('DONE\n\n');