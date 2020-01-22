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
        gamma_down_m_mj_up_o(i_m,i_mj,1) = max([0, min([(m - mj + 1) / 2, 1])]);
%         gamma_down_m_mj_up_o(i_m,i_mj,1) = 0.5 * (1 + erf((m - mj) / (sqrt(2) * ne_param.gamma_s)));
%         gamma_down_m_mj_up_o(i_m,i_mj,1) = 1 / (1 + exp(-(m - mj) / ne_param.gamma_s));
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
            beta_down_k_m_up_mb(i_k,i_m,i_mb) = max([0, 1 - abs(mb - m)]);
%             beta_down_k_m_up_mb(i_k,i_m,i_mb) = exp(-(mb - m)^2 / (2 * ne_param.beta_s^2));
%             beta_down_k_m_up_mb(i_k,i_m,i_mb) = exp(-(mb - m) / ne_param.beta_s) / ((1 + exp(-(mb - m) / ne_param.beta_s))^2);
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
upsilon_down_k_mb_kj_mj_o_up_kn = permute(squeeze(dot2(permute(epsilon_down_k_mb_kj_mbj_o_up_kn, [1 2 5 6 3 4]), permute(beta_down_k_m_up_mb, [1 3 2]), 6, 2)), [1 2 5 6 3 4]);
clearvars epsilon_down_k_mb_kj_mbj_o_up_kn;
phi_down_k_m_kj_mj_o_up_kn = permute(squeeze(dot2(permute(upsilon_down_k_mb_kj_mj_o_up_kn, [3 4 5 6 1 2]), permute(beta_down_k_m_up_mb, [1 3 2]), 6, 2)), [5 6 1 2 3 4]);
clearvars beta_down_k_m_up_mb upsilon_down_k_mb_kj_mj_o_up_kn;
kappa_down_k_m_kj_mj_up_kn = permute(dot2(permute(phi_down_k_m_kj_mj_o_up_kn, [1 3 6 2 4 5]), gamma_down_m_mj_up_o, 6, 3), [1 4 2 5 3]);
clearvars gamma_down_m_mj_up_o phi_down_k_m_kj_mj_o_up_kn;
psi_down_u_k_m_kj_mj_up_un_kn = permute(reshape(outer(reshape(ne_param.mu_down_u_up_un, [], 1), kappa_down_k_m_kj_mj_up_kn), [ne_param.num_U, ne_param.num_U, size(kappa_down_k_m_kj_mj_up_kn)]), [1 3 4 5 6 2 7]);
clearvars kappa_down_k_m_kj_mj_up_kn;

%% From game tensors to cells of 2 dimensional matrices
C = cell(ne_param.num_X, 1);
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        C{i_x} = squeeze(c_down_u_m_mj(i_u,:,:));
    end
end
clearvars c_down_u_m_mj;

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
clearvars psi_down_u_k_m_kj_mj_up_un_kn;

%% Optimization variables
pi = sdpvar(ne_param.num_X, ne_param.num_M);
d = sdpvar(ne_param.num_X, 1);
a = sdpvar(ne_param.num_M, 1);
q = sdpvar(ne_param.num_X, 1);
b = sdpvar(ne_param.num_X * ne_param.num_M, 1);
t = sdpvar(ne_param.num_X);
v = sdpvar(ne_param.num_X, 1);

%% Constraints
constraints = a == pi.' * d;
% for i_m = 1 : ne_param.num_M
%     constraints = [constraints; a(i_m) == d.' * pi(:,i_m)];
% end
for i_x = 1 : ne_param.num_X
    constraints = [constraints; q(i_x) == pi(i_x,:) * C{i_x} * a];
end
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
constraints = [constraints; v == q + ne_param.alpha * t * v];
constraints = [constraints; d == t.' * d];
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        i_adm_m = find(ne_param.M <= ne_param.K(i_k));
        constraints = [constraints; pi(i_x,i_adm_m) >= -eps];
        constraints = [constraints; pi(i_x,i_adm_m(end)+1:end) == 0];
    end
end
constraints = [constraints; sum(pi, 2) == ones(ne_param.num_X, 1)];
constraints = [constraints; d >= -eps];
constraints = [constraints; sum(d) == 1];

%% Objective
objective = d.' * v;

%% Initial guess
% Try bid 1 if urgent policy
pi_0 = zeros(ne_param.num_X, ne_param.num_M);
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        if ne_param.U(i_u) == 0 || i_k < 2
            pi_0(i_x,1) = 1;
        else
            pi_0(i_x,2) = 1;
        end
    end
end

% % Try stationary distribution all at k_max
% d_0 = zeros(ne_param.num_X, 1);
% for i_u = 1 : ne_param.num_U
%     i_u_base = (i_u - 1) * ne_param.num_K;
%     d_0(i_u_base+ne_param.num_K) = 0.5;
% end

% Try uniform distribtuion (leading to k_ave = k_max / 2)
d_0 = rand(ne_param.num_X);
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

% Get initial guess of a
a_0 = pi_0.' * d_0;

% Get initial guess of q
q_0 = zeros(ne_param.num_X, 1);
for i_x = 1 : ne_param.num_X
    q_0(i_x) = pi_0(i_x,:) * C{i_x} * a_0;
end

% Get initial guess of v
v_0 = (eye(ne_param.num_X) - ne_param.alpha * t_0) \ q_0;

assign(pi, pi_0);
assign(d, d_0);
assign(a, a_0);
assign(q, q_0);
assign(b, b_0);
for i_x = 1 : ne_param.num_X
    assign(t(i_x,:), t_0(i_x,:));
end
assign(v, v_0);

%% Solve
options = sdpsettings('solver', 'ipopt', 'usex0', 1);
% options.fmincon.FinDiffType = 'central';
% options.fmincon.FinDiffRelStep = eps^(1/5);
optimize(constraints, objective, options);

%% Get solution
pi_val = double(pi);
d_val = double(d);
v_val = double(v);
a_val = double(a);
q_val = double(q);
b_val = double(b);
t_val = double(t);
obj_val = double(objective);