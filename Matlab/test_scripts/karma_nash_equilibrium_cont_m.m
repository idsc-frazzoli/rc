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

%% NE policy guess
ne_m_down_u_k = rand(ne_param.num_U, ne_param.num_K);
ne_m_down_u_k(:,1) = zeros(ne_param.num_U, 1);
ne_m_down_u_k(:,2:end) = ne_m_down_u_k(:,2:end) + ne_param.K(1:end-1).';

%% delta(m) & omega tensor
delta_down_u_k_uj_kj = cell(ne_param.num_U, ne_param.num_K, ne_param.num_U, ne_param.num_K);
omega_down_u_k_uj_kj = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_U, ne_param.num_K);
for i_u = 1 : ne_param.num_U
    u = ne_param.U(i_u);
    for i_k = 1 : ne_param.num_K
        ne_m = ne_m_down_u_k(i_u,i_k);
        for i_uj = 1 : ne_param.num_U
            for i_kj = 1 : ne_param.num_K
                fprintf('delta(m): i_u %d i_k %d i_uj %d i_kj %d\n', i_u, i_k, i_uj, i_kj);
                mj = ne_m_down_u_k(i_uj,i_kj);
                delta_down_u_k_uj_kj{i_u,i_k,i_uj,i_kj} = ...
                    @(m) c_down_u(u, m, mj);
                
                omega_down_u_k_uj_kj(i_u,i_k,i_uj,i_kj) = ...
                    delta_down_u_k_uj_kj{i_u,i_k,i_uj,i_kj}(ne_m);
            end
        end
    end
end

%% lambda(m) & sigma tensors
lambda_down_u_k_uj_kj_up_un_kn = cell(ne_param.num_U,ne_param.num_K, ne_param.num_U, ne_param.num_K, ne_param.num_U, ne_param.num_K);
sigma_down_u_k_uj_kj_up_un_kn = zeros(ne_param.num_U,ne_param.num_K, ne_param.num_U, ne_param.num_K, ne_param.num_U, ne_param.num_K);
for i_u = 1 : ne_param.num_U
    for i_k = 1 : ne_param.num_K
        k = ne_param.K(i_k);
        ne_m = ne_m_down_u_k(i_u,i_k);
        for i_uj = 1 : ne_param.num_U
            for i_kj = 1 : ne_param.num_K
                kj = ne_param.K(i_kj);
                mj = ne_m_down_u_k(i_uj,i_kj);
                for i_un = 1 : ne_param.num_U
                    mu = ne_param.mu_down_u_up_un(i_u,i_un);
                    for i_kn = 1 : ne_param.num_K
                        fprintf('lambda(m): i_u %d i_k %d i_uj %d i_kj %d i_un %d i_kn %d\n', i_u, i_k, i_uj, i_kj, i_un, i_kn);
                        kn = ne_param.K(i_kn);
                        lambda_down_u_k_uj_kj_up_un_kn{i_u,i_k,i_uj,i_kj,i_un,i_kn} = ...
                            @(m) (mu * phi_down_k_kj_up_kn(k, kj, kn, m, mj, ne_param.k_max));
                        
                        sigma_down_u_k_uj_kj_up_un_kn(i_u,i_k,i_uj,i_kj,i_un,i_kn) = ...
                            lambda_down_u_k_uj_kj_up_un_kn{i_u,i_k,i_uj,i_kj,i_un,i_kn}(ne_m);
                    end
                end
            end
        end
    end
end

%% Stationary distribution
i_kave = find(ne_param.K == ne_param.k_ave);
if ne_param.k_ave * 2 <= ne_param.k_max
    i_kave2 = find(ne_param.K == ne_param.k_ave * 2);
    d_up_k_init = [1 / i_kave2 * ones(i_kave2, 1); zeros(ne_param.num_K - i_kave2, 1)];
else
    d_up_k_init = 1 / ne_param.num_K * ones(ne_param.num_K, 1);
    K_small = ne_param.k_min : ne_param.k_ave - 1;
    K_big = ne_param.k_ave + 1 : ne_param.k_max;
    num_K_small = length(K_small);
    num_K_big = length(K_big);
    delta_constant = sum(K_small) / num_K_small - sum(K_big) / num_K_big;
    delta_k_ave = ne_param.k_ave - ne_param.K.' * d_up_k_init;
    delta_p = delta_k_ave / delta_constant;
    d_up_k_init(1:i_kave-1) = d_up_k_init(1:i_kave-1) + delta_p / num_K_small;
    d_up_k_init(i_kave+1:end) = d_up_k_init(i_kave+1:end) - delta_p / num_K_big;
end
d_up_u_init = ne_param.p_U;
d_up_u_k = d_up_u_init * d_up_k_init.';
d_up_u_k_next = d_up_u_k;
T_down_u_k_up_un_kn = squeeze(dot2(reshape(d_up_u_k_next, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
d_up_u_k_next_next = squeeze(dot2(reshape(T_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(d_up_u_k_next, [], 1), 1, 1));
d_up_u_k_next_next = d_up_u_k_next_next / sum(d_up_u_k_next_next(:));
d_error = norm(d_up_u_k_next - d_up_u_k_next_next, inf);
d_up_u_k_next = d_up_u_k_next_next;
num_d_iter = 0;
while d_error > ne_param.D_tol && num_d_iter < ne_param.D_max_iter
    T_down_u_k_up_un_kn = squeeze(dot2(reshape(d_up_u_k_next, [], 1), reshape(sigma_down_u_k_uj_kj_up_un_kn, ne_param.num_U, ne_param.num_K, [], ne_param.num_U, ne_param.num_K), 1, 3));
    d_up_u_k_next_next = squeeze(dot2(reshape(T_down_u_k_up_un_kn, [], ne_param.num_U, ne_param.num_K), reshape(d_up_u_k_next, [], 1), 1, 1));
    d_up_u_k_next_next = d_up_u_k_next_next / sum(d_up_u_k_next_next(:));
    d_error = norm(d_up_u_k_next - d_up_u_k_next_next, inf);
    d_up_u_k_next = d_up_u_k_next_next;
    num_d_iter = num_d_iter + 1;
end
% Apply momentum
d_up_u_k = ne_param.D_tau * d_up_u_k_next + (1 - ne_param.D_tau) * d_up_u_k;

%% q tensor
q_down_u_k = dot2(reshape(d_up_u_k, [], 1), reshape(omega_down_u_k_uj_kj, ne_param.num_U, ne_param.num_K, []), 1, 3);

%% Inifinite horizon cost
V_down_u_k = zeros(ne_param.num_U, ne_param.num_K);
V_down_u_k_next = q_down_u_k + ne_param.alpha * dot2(reshape(T_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(V_down_u_k, [], 1), 3, 1);
V_error = norm(V_down_u_k - V_down_u_k_next, inf);
V_down_u_k = V_down_u_k_next;
num_V_iter = 0;
while V_error > ne_param.V_tol && num_V_iter < ne_param.V_max_iter
    V_down_u_k_next = q_down_u_k + ne_param.alpha * dot2(reshape(T_down_u_k_up_un_kn, ne_param.num_U, ne_param.num_K, []), reshape(V_down_u_k, [], 1), 3, 1);
    V_error = norm(V_down_u_k - V_down_u_k_next, inf);
    V_down_u_k = V_down_u_k_next;
    num_V_iter = num_V_iter + 1;
end

%% zeta(m) tensor
zeta_down_u_k = cell(ne_param.num_U, ne_param.num_K);
q_down_u_k_check = zeros(ne_param.num_U, ne_param.num_K);
for i_u = 1 : ne_param.num_U
    for i_k = 1 : ne_param.num_K
        fprintf('zeta(m): i_u %d i_k %d\n', i_u, i_k);
        zeta_down_u_k{i_u,i_k} = @(m) zeta_down_u_k_fun(i_u, i_k, ne_param.num_U, ne_param.num_K, d_up_u_k, delta_down_u_k_uj_kj, m);
        
        ne_m = ne_m_down_u_k(i_u,i_k);
        q_down_u_k_check(i_u,i_k) = ...
            zeta_down_u_k{i_u,i_k}(ne_m);
    end
end

%% chi(m) tensor
chi_down_u_k_up_un_kn = cell(ne_param.num_U, ne_param.num_K, ne_param.num_U, ne_param.num_K);
T_down_u_k_up_un_kn_check = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_U, ne_param.num_K);
for i_u = 1 : ne_param.num_U
    for i_k = 1 : ne_param.num_K
        ne_m = ne_m_down_u_k(i_u,i_k);
        for i_un = 1 : ne_param.num_U
            for i_kn = 1 : ne_param.num_K
                fprintf('chi(m): i_u %d i_k %d i_un %d i_kn %d\n', i_u, i_k, i_un, i_kn);
                chi_down_u_k_up_un_kn{i_u,i_k,i_un,i_kn} = @(m) chi_down_u_k_up_un_kn_fun(i_u, i_k, i_un, i_kn, ne_param.num_U, ne_param.num_K, d_up_u_k, lambda_down_u_k_uj_kj_up_un_kn, m);
                
                T_down_u_k_up_un_kn_check(i_u,i_k,i_un,i_kn) = ...
                    chi_down_u_k_up_un_kn{i_u,i_k,i_un,i_kn}(ne_m);
            end
        end
    end
end

%% upsilon(m) tensor
upsilon_down_u_k = cell(ne_param.num_U, ne_param.num_K);
for i_u = 1 : ne_param.num_U
    for i_k = 1 : ne_param.num_K
        fprintf('upsilon(m): i_u %d i_k %d\n', i_u, i_k);
        upsilon_down_u_k{i_u,i_k} = @(m) upsilon_down_u_k_fun(i_u, i_k, ne_param.num_U, ne_param.num_K, V_down_u_k, chi_down_u_k_up_un_kn, m);
    end
end

%% rho(m) tensor
rho_down_u_k = cell(ne_param.num_U, ne_param.num_K);
V_down_u_k_check = zeros(ne_param.num_U, ne_param.num_K);
for i_u = 1 : ne_param.num_U
    for i_k = 1 : ne_param.num_K
        fprintf('rho(m): i_u %d i_k %d\n', i_u, i_k);
        rho_down_u_k{i_u,i_k} = @(m) zeta_down_u_k{i_u,i_k}(m) + ne_param.alpha * upsilon_down_u_k{i_u,i_k}(m);
        
        ne_m = ne_m_down_u_k(i_u,i_k);
        V_down_u_k_check(i_u,i_k) = ...
            rho_down_u_k{i_u,i_k}(ne_m);
    end
end

%% Functions
function c = c_down_u(u, m, mj)
    if m  > mj
        c = 0;
    elseif m < mj
        c = u;
    else
        c = 0.5 * u;
    end
end

function p = phi_down_k_kj_up_kn(k, kj, kn, m, mj, k_max)
    z = floor(m);
    r = m - z;
    zj = floor(mj);
    rj = mj - zj;
    min_z_k_max_kj = min([z, k_max - kj]);
    min_z_1_k_max_kj = min([z + 1, k_max - kj]);
    min_k_zj_k_max = min([k + zj, k_max]);
    min_k_zj_1_k_max = min([k + zj + 1, k_max]);
    
    p = 0;
    if m > mj
        if kn == k - min_z_k_max_kj
            p = p + 1 - r;
        end
        if kn == k - min_z_1_k_max_kj
            p = p + r;
        end
    elseif m < mj
        if kn == min_k_zj_k_max
            p = p + 1 - rj;
        end
        if kn == min_k_zj_1_k_max
            p = p + rj;
        end
    else
        if kn == k - min_z_k_max_kj
            p = p + 0.5 * (1 - r);
        end
        if kn == k - min_z_1_k_max_kj
            p = p + 0.5 * r;
        end
        if kn == min_k_zj_k_max
            p = p + 0.5 * (1 - rj);
        end
        if kn == min_k_zj_1_k_max
            p = p + 0.5 * rj;
        end
    end
end

function c = zeta_down_u_k_fun(i_u, i_k, num_U, num_K, d_up_u_k, delta_down_u_k_uj_kj, m)
    c = 0;
    for i_uj = 1 : num_U
        for i_kj = 1 : num_K
            c = c + d_up_u_k(i_uj,i_kj) * delta_down_u_k_uj_kj{i_u,i_k,i_uj,i_kj}(m);
        end
    end
end

function p = chi_down_u_k_up_un_kn_fun(i_u, i_k, i_un, i_kn, num_U, num_K, d_up_u_k, lambda_down_u_k_uj_kj_up_un_kn, m)
    p = 0;
    for i_uj = 1 : num_U
        for i_kj = 1 : num_K
            p = p + d_up_u_k(i_uj,i_kj) * lambda_down_u_k_uj_kj_up_un_kn{i_u,i_k,i_uj,i_kj,i_un,i_kn}(m);
        end
    end
end

function c = upsilon_down_u_k_fun(i_u, i_k, num_U, num_K, V_down_u_k, chi_down_u_k_up_un_kn, m)
    c = 0;
    for i_un = 1 : num_U
        for i_kn = 1 : num_K
            c = c + V_down_u_k(i_un,i_kn) * chi_down_u_k_up_un_kn{i_u,i_k,i_un,i_kn}(m);
        end
    end
end