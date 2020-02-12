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
% [pi; pi^1; ...; pi^Nm; d; b; t; t^1; ...; t^Nx; a; q]
pi_ind = 0;
pi_size = ne_param.num_X * ne_param.num_M;
pi_m_ind = pi_ind + pi_size;
d_ind = pi_m_ind + pi_size;
d_size = ne_param.num_X;
b_ind = d_ind + d_size;
b_size = ne_param.num_X * ne_param.num_M;
t_ind = b_ind + b_size;
t_size = ne_param.num_X * ne_param.num_X;
t_xn_ind = t_ind + t_size;
a_ind = t_xn_ind + t_size;
a_size = ne_param.num_M;
q_ind = a_ind + a_size;
q_size = ne_param.num_X;

num_vars = 2 * pi_size + d_size + b_size + 2 * t_size + a_size + q_size;

%% Quadratic cost matrix
Q = zeros(num_vars);
Q(d_ind+1:d_ind+d_size,q_ind+1:q_ind+q_size) = 0.5 * eye(d_size,q_size);
Q(q_ind+1:q_ind+q_size,d_ind+1:d_ind+d_size) = 0.5 * eye(q_size,d_size);

grb_model.Q = sparse(Q);

%% Bound constraints
% Everything is positive in program so Gurobi default bounds (0,inf) is good

%% Linear equality constraints
% Initialize constraints
A = [];
b = [];

% pi is a probability distribution for each state. Messages that are higher
% than karma for given state are disallowed
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        i_x_base = (i_x - 1) * ne_param.num_M;
        i_good_m = find(ne_param.M <= ne_param.K(i_k));
        num_good_m = length(i_good_m);
        A_temp = zeros(1, num_vars);
        A_temp(:,pi_ind+i_x_base+i_good_m) = ones(1, num_good_m);
        b_temp = 1;
        A = [A; A_temp];
        b = [b; b_temp];
        
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

% d is a probability distribution with average karma k_ave
A_temp = zeros(2, num_vars);
A_temp(1,d_ind+1:d_ind+d_size) = ones(1, d_size);
A_temp(2,d_ind+1:d_ind+d_size) = repmat(ne_param.K, ne_param.num_U, 1).';
b_temp = [1; ne_param.k_ave];
A = [A; A_temp];
b = [b; b_temp];

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
% Initialize constraints
quadcon_i = 1;

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

% Constraints for a
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

%% Initial guess
% Try bid 1 if urgent policy
pi_mat_0 = zeros(ne_param.num_X, ne_param.num_M);
for i_u = 1 : ne_param.num_U
    i_u_base = (i_u - 1) * ne_param.num_K;
    for i_k = 1 : ne_param.num_K
        i_x = i_u_base + i_k;
        if ne_param.U(i_u) == 0 || ne_param.K(i_k) == 0
            pi_mat_0(i_x,1) = 1;
        else
            pi_mat_0(i_x,2) = 1;
        end
    end
end
pi_0 = reshape(pi_mat_0.', [], 1);
pi_m_0 = reshape(pi_mat_0, [], 1);

% % Try stationary distribution all at k_max
% d_0 = zeros(ne_param.num_X, 1);
% for i_u = 1 : ne_param.num_U
%     i_u_base = (i_u - 1) * ne_param.num_K;
%     d_0(i_u_base+ne_param.num_K) = 0.5;
% end

% Find initial distribution corresponding to initial policy and with
% required k_ave
d_u_0 = ne_func.stat_dist(ne_param.mu_down_u_up_un);
i_kave = find(ne_param.K == ne_param.k_ave);
if ne_param.k_ave * 2 <= ne_param.k_max
    i_kave2 = find(ne_param.K == ne_param.k_ave * 2);
    d_k_0 = [1 / i_kave2 * ones(i_kave2, 1); zeros(ne_param.num_K - i_kave2, 1)];
elseif ne_param.k_ave >= ne_param.k_max
    d_k_0 = zeros(ne_param.num_K, 1);
    d_k_0(end) = 1;
else
    d_k_0 = 1 / ne_param.num_K * ones(ne_param.num_K, 1);
    K_small = ne_param.k_min : ne_param.k_ave - 1;
    K_big = ne_param.k_ave + 1 : ne_param.k_max;
    num_K_small = length(K_small);
    num_K_big = length(K_big);
    delta_constant = sum(K_small) / num_K_small - sum(K_big) / num_K_big;
    delta_k_ave = ne_param.k_ave - ne_param.K.' * d_k_0;
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
    b_0(start_i:end_i) = d_0(i_x) * pi_mat_0(i_x,:).';
end

% Get initial guess of t
t_mat_0 = zeros(ne_param.num_X);
for i_x = 1 : ne_param.num_X
    for i_xn = 1 : ne_param.num_X
        t_mat_0(i_x,i_xn) = pi_mat_0(i_x,:) * Psi{i_x,i_xn} * b_0;
    end
end

% Update initial guesses d_0, b_0, t_mat_0 until convergence
while norm(d_0 - t_mat_0.' * d_0, inf) > 1e-12
    d_0 = t_mat_0.' * d_0;
    d_0 = d_0 / sum(d_0);
    
    for i_x = 1 : ne_param.num_X
        start_i = (i_x - 1) * ne_param.num_M + 1;
        end_i = i_x * ne_param.num_M;
        b_0(start_i:end_i) = d_0(i_x) * pi_mat_0(i_x,:).';
    end
    
    for i_x = 1 : ne_param.num_X
        for i_xn = 1 : ne_param.num_X
            t_mat_0(i_x,i_xn) = pi_mat_0(i_x,:) * Psi{i_x,i_xn} * b_0;
        end
    end
end
t_0 = reshape(t_mat_0.', [], 1);
t_xn_0 = reshape(t_mat_0, [], 1);

% Get initial guess of a
a_0 = pi_mat_0.' * d_0;

% Get initial guess of q
q_0 = zeros(ne_param.num_X, 1);
for i_x = 1 : ne_param.num_X
    q_0(i_x) = pi_mat_0(i_x,:) * C{i_x} * a_0;
end

% Assign to start vector
grb_model.start = [pi_0; pi_m_0; d_0; b_0; t_0; t_xn_0; a_0; q_0];

%% Optimization parameters
grb_params.NonConvex = 2;
% grb_params.MIPFocus = 1;

%% Now hope for the best and solve
result = gurobi(grb_model, grb_params);

%% Get optimization variable values
pi = result.x(pi_ind+1:pi_ind+pi_size);
pi_mat = reshape(pi, ne_param.num_M, []);
pi_m = result.x(pi_m_ind+1:pi_m_ind+pi_size);
pi_m_mat = reshape(pi_m, ne_param.num_X, []);
d = result.x(d_ind+1:d_ind+d_size);
b = result.x(b_ind+1:b_ind+b_size);
t = result.x(t_ind+1:t_ind+t_size);
t_mat = reshape(t, ne_param.num_X, []);
t_xn = result.x(t_xn_ind+1:t_xn_ind+t_size);
t_xn_mat = reshape(t_xn, ne_param.num_X, []);
a = result.x(a_ind+1:a_ind+a_size);
q = result.x(q_ind+1:q_ind+q_size);
obj = d.' * q;