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

%% Optimization variables:
% [d; v; pi; pi^1; ...; pi^Nm; a; q; b; t; t^1; ...; t^Nx]
d_ind = 0;
d_size = ne_param.num_X;
v_ind = d_ind + d_size;
v_size = ne_param.num_X;
pi_ind = v_ind + v_size;
pi_size = ne_param.num_X * ne_param.num_M;
pi_m_ind = pi_ind + pi_size;
a_ind = pi_m_ind + pi_size;
a_size = ne_param.num_M;
q_ind = a_ind + a_size;
q_size = ne_param.num_X;
b_ind = q_ind + q_size;
b_size = ne_param.num_X * ne_param.num_M;
t_ind = b_ind + b_size;
t_size = ne_param.num_X * ne_param.num_X;
t_xn_ind = t_ind + t_size;
num_vars = d_size + v_size + 2 * pi_size + a_size + q_size + b_size + 2 * t_size;

%% Quadratic cost matrix
Q = zeros(num_vars);
Q(d_ind+1:d_ind+d_size,v_ind+1:v_ind+v_size) = 0.5 * eye(d_size,v_size);
Q(v_ind+1:v_ind+v_size,d_ind+1:d_ind+d_size) = 0.5 * eye(v_size,d_size);

grb_model.Q = sparse(Q);

%% Bound constraints
grb_model.lb = -inf * ones(num_vars, 1);
% d and pi are probabilities
grb_model.lb(d_ind+1:d_ind+d_size) = 0;
grb_model.lb(pi_ind+1:pi_ind+pi_size) = 0;

grb_model.ub = inf * ones(num_vars, 1);

%% Linear equality constraints
% d is a probability distribution
A = zeros(1, num_vars);
A(:,d_ind+1:d_ind+d_size) = ones(1, d_size);
b = 1;

% pi is a probability distribution for each state
for i_x = 1 : ne_param.num_X
    i_x_base = (i_x - 1) * ne_param.num_M;
    A_temp = zeros(1, num_vars);
    A_temp(:,pi_ind+i_x_base+1:pi_ind+i_x_base+ne_param.num_M) = ones(1, ne_param.num_M);
    b_temp = 1;
    A = [A; A_temp];
    b = [b; b_temp];
end

% Unallowed messages that are higher than karma for given state
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        i_x_base = (i_x - 1) * ne_param.num_M;
        i_bad_m = find(ne_param.M > ne_param.K(i_k));
        num_bad_m = length(i_bad_m);
        if num_bad_m > 0
            A_temp = zeros(num_bad_m, num_vars);
            A_temp(:,pi_ind+i_x_base+i_bad_m) = eye(num_bad_m);
            b_temp = zeros(num_bad_m, 1);
            A = [A; A_temp];
            b = [b; b_temp];
        end
    end
end

% Constraints for transpose of policy matrix
b_temp = zeros(ne_param.num_X, 1);
for i_m = 1 : ne_param.num_M
    i_m_base = (i_m - 1) * ne_param.num_X;
    A_temp = zeros(ne_param.num_X, num_vars);
    for i_x = 1 : ne_param.num_X
        i_x_base = (i_x - 1) * ne_param.num_M;
        A_temp(i_x,pi_ind+i_x_base+i_m) = 1;
    end
    A_temp(:,pi_m_ind+i_m_base+1:pi_m_ind+i_m_base+ne_param.num_X) = -eye(ne_param.num_X);
    A = [A; A_temp];
    b = [b; b_temp];
end

% Constraints for transpose of transformation matrix
b_temp = zeros(ne_param.num_X, 1);
for i_xn = 1 : ne_param.num_X
    i_xn_base = (i_xn - 1) * ne_param.num_X;
    A_temp = zeros(ne_param.num_X, num_vars);
    for i_x = 1 : ne_param.num_X
        i_x_base = (i_x - 1) * ne_param.num_X;
        A_temp(i_x,t_ind+i_x_base+i_xn) = 1;
    end
    A_temp(:,t_xn_ind+i_xn_base+1:t_xn_ind+i_xn_base+ne_param.num_X) = -eye(ne_param.num_X);
    A = [A; A_temp];
    b = [b; b_temp];
end

grb_model.A = sparse(A);
grb_model.rhs = b;
grb_model.sense = repmat('=', 1, size(A,1));

%% Quadratic equality constraints
% Constraints for a
quadcon_i = 1;
for i_m = 1 : ne_param.num_M
    i_m_base = (i_m - 1) * ne_param.num_X;
    Qc = zeros(num_vars);
    Qc(d_ind+1:d_ind+d_size,pi_m_ind+i_m_base+1:pi_m_ind+i_m_base+ne_param.num_X) = 0.5 * eye(d_size,ne_param.num_X);
    Qc(pi_m_ind+i_m_base+1:pi_m_ind+i_m_base+ne_param.num_X,d_ind+1:d_ind+d_size) = 0.5 * eye(ne_param.num_X,d_size);
    q = zeros(num_vars, 1);
    q(a_ind+i_m) = -1;
    beta = 0;
    
    grb_model.quadcon(quadcon_i).Qc = sparse(Qc);
    grb_model.quadcon(quadcon_i).q = sparse(q);
    grb_model.quadcon(quadcon_i).rhs = beta;
    grb_model.quadcon(quadcon_i).sense = '=';
    
    quadcon_i = quadcon_i + 1;
end

% Constraints for q
for i_x = 1 : ne_param.num_X
    i_x_base = (i_x - 1) * ne_param.num_M;
    Qc = zeros(num_vars);
    Qc(pi_ind+i_x_base+1:pi_ind+i_x_base+ne_param.num_M,a_ind+1:a_ind+a_size) = 0.5 * C{i_x};
    Qc(a_ind+1:a_ind+a_size,pi_ind+i_x_base+1:pi_ind+i_x_base+ne_param.num_M) = 0.5 * C{i_x}.';
    q = zeros(num_vars, 1);
    q(q_ind+i_x) = -1;
    beta = 0;
    
    grb_model.quadcon(quadcon_i).Qc = sparse(Qc);
    grb_model.quadcon(quadcon_i).q = sparse(q);
    grb_model.quadcon(quadcon_i).rhs = beta;
    grb_model.quadcon(quadcon_i).sense = '=';
    
    quadcon_i = quadcon_i + 1;
end

% Constraints for b
for i_x = 1 : ne_param.num_X
    i_x_base = (i_x - 1) * ne_param.num_M;
    for i_m = 1 : ne_param.num_M
        Qc = zeros(num_vars);
        Qc(d_ind+i_x,pi_ind+i_x_base+i_m) = 0.5;
        Qc(pi_ind+i_x_base+i_m,d_ind+i_x) = 0.5;
        q = zeros(num_vars, 1);
        q(b_ind+i_x_base+i_m) = -1;
        beta = 0;

        grb_model.quadcon(quadcon_i).Qc = sparse(Qc);
        grb_model.quadcon(quadcon_i).q = sparse(q);
        grb_model.quadcon(quadcon_i).rhs = beta;
        grb_model.quadcon(quadcon_i).sense = '=';

        quadcon_i = quadcon_i + 1;
    end
end

% Constraints for t
for i_x = 1 : ne_param.num_X
    i_x_base_pi = (i_x - 1) * ne_param.num_M;
    i_x_base_t = (i_x - 1) * ne_param.num_X;
    for i_xn = 1 : ne_param.num_X
        Qc = zeros(num_vars);
        Qc(pi_ind+i_x_base_pi+1:pi_ind+i_x_base_pi+ne_param.num_M,b_ind+1:b_ind+b_size) = 0.5 * Psi{i_x,i_xn};
        Qc(b_ind+1:b_ind+b_size,pi_ind+i_x_base_pi+1:pi_ind+i_x_base_pi+ne_param.num_M) = 0.5 * Psi{i_x,i_xn}.';
        q = zeros(num_vars, 1);
        q(t_ind+i_x_base_t+i_xn) = -1;
        beta = 0;

        grb_model.quadcon(quadcon_i).Qc = sparse(Qc);
        grb_model.quadcon(quadcon_i).q = sparse(q);
        grb_model.quadcon(quadcon_i).rhs = beta;
        grb_model.quadcon(quadcon_i).sense = '=';

        quadcon_i = quadcon_i + 1;
    end
end

% Constraints for v
for i_x = 1 : ne_param.num_X
    i_x_base = (i_x - 1) * ne_param.num_X;
    Qc = zeros(num_vars);
    Qc(v_ind+1:v_ind+v_size,t_ind+i_x_base+1:t_ind+i_x_base+ne_param.num_X) = 0.5 * ne_param.alpha * eye(ne_param.num_X);
    Qc(t_ind+i_x_base+1:t_ind+i_x_base+ne_param.num_X,v_ind+1:v_ind+v_size) = 0.5 * ne_param.alpha * eye(ne_param.num_X);
    q = zeros(num_vars, 1);
    q(v_ind+i_x) = -1;
    q(q_ind+i_x) = 1;
    beta = 0;
    
    grb_model.quadcon(quadcon_i).Qc = sparse(Qc);
    grb_model.quadcon(quadcon_i).q = sparse(q);
    grb_model.quadcon(quadcon_i).rhs = beta;
    grb_model.quadcon(quadcon_i).sense = '=';
    
    quadcon_i = quadcon_i + 1;
end

% Constraints for d
for i_xn = 1 : ne_param.num_X
    i_xn_base = (i_xn - 1) * ne_param.num_X;
    Qc = zeros(num_vars);
    Qc(d_ind+1:d_ind+d_size,t_xn_ind+i_xn_base+1:t_xn_ind+i_xn_base+ne_param.num_X) = 0.5 * eye(ne_param.num_X);
    Qc(t_xn_ind+i_xn_base+1:t_xn_ind+i_xn_base+ne_param.num_X,d_ind+1:d_ind+d_size) = 0.5 * eye(ne_param.num_X);
    q = zeros(num_vars, 1);
    q(d_ind+i_xn) = -1;
    beta = 0;
    
    grb_model.quadcon(quadcon_i).Qc = sparse(Qc);
    grb_model.quadcon(quadcon_i).q = sparse(q);
    grb_model.quadcon(quadcon_i).rhs = beta;
    grb_model.quadcon(quadcon_i).sense = '=';
    
    quadcon_i = quadcon_i + 1;
end

%% Optimization parameters
grb_params.NonConvex = 2;
% grb_params.MIPFocus = 1;

%% Now hope for the best and solve
result = gurobi(grb_model, grb_params);

%% Get optimization variable values
d = result.x(d_ind+1:d_ind+d_size);
v = result.x(v_ind+1:v_ind+v_size);
pi = result.x(pi_ind+1:pi_ind+pi_size);
pi_mat = reshape(pi, ne_param.num_M, []);
pi_m = result.x(pi_m_ind+1:pi_m_ind+pi_size);
pi_m_mat = reshape(pi_m, ne_param.num_X, []);
a = result.x(a_ind+1:a_ind+a_size);
q = result.x(q_ind+1:q_ind+q_size);
b = result.x(b_ind+1:b_ind+b_size);
t = result.x(t_ind+1:t_ind+t_size);
t_mat = reshape(t, ne_param.num_X, []);
t_xn = result.x(t_xn_ind+1:end);
t_xn_mat = reshape(t_xn, ne_param.num_X, []);