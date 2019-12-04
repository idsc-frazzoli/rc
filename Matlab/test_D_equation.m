clc;
clear all;
close all;

%% Create random T matrices that sum to 1
rng(0);

Nx = 2;

T = cell(Nx, 1);
T_sum = 0;
for i_x = 1 : Nx
    T{i_x} = rand(Nx);
    T_sum = T_sum + T{i_x};
end
for i_x = 1 : Nx
    T{i_x} = T{i_x} ./ T_sum;
end
T_sum = 0;
for i_x = 1 : Nx
    T_sum = T_sum + T{i_x};
end

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
    else
        D_init(:,i) = rand(Nx, 1);
        D_init(:,i) = D_init(:,i) / sum(D_init(:,i));
    end
    D_curr = D_init(:,i);
    D_next = zeros(Nx, 1);
    for i_x = 1 : Nx
        D_next(i_x) = D_curr.' * T{i_x} * D_curr;
    end
    while(norm(D_next - D_curr, inf) > 1e-4)
        D_curr = D_next;
        for i_x = 1 : Nx
            D_next(i_x) = D_curr.' * T{i_x} * D_curr;
        end
    end
    D(:,i) = D_next;
end
sensitivity = norm(max(D, 2) - min(D, 2), inf);

fprintf('DONE ITERATIVE\n\n');

%% QCQP solution
D_opt = sdpvar(Nx, 1);
F = ones(1, Nx) * D_opt == 1;   % D_opt entries sum to 1
F = [F; eye(Nx) * D_opt >= zeros(Nx, 1)];  % D_opt entries are non-negative
for i_x = 1 : Nx
    e_vec = zeros(Nx, 1);
    e_vec(i_x) = 1;
    F = [F; e_vec.' * D_opt == D_opt.' * T{i_x} * D_opt];
end
options = sdpsettings('solver', 'fmincon', 'verbose', 1);
optimize(F, [], options);
D_sol = value(D_opt);

sensitivity_opt = norm(max([D, D_sol], 2) - min([D, D_sol], 2), inf);

fprintf('DONE QCQP\n\n');

%% Kronecker products
T_all = T{1};
e_all = zeros(Nx, 1);
e_all(1) = 1;
for i_x = 2 : Nx
    T_all = blkdiag(T_all, T{i_x});
    e_vec = zeros(Nx, 1);
    e_vec(i_x) = 1;
    e_all = [e_all; e_vec];
end
D_mat = kron(eye(Nx), D_sol);
D_kron = D_mat.' * T_all * D_mat * ones(Nx, 1);



fprintf('DONE\n\n');