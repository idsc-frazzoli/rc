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
ne_m_down_u_k = zeros(ne_param.num_U, ne_param.num_K);

%% Symbolic m
m_down_u_k = sym('m', [ne_param.num_U ne_param.num_K]);

%% delta(m) tensor
delta_down_u_k_uj_kj = sym('delta', [ne_param.num_U ne_param.num_K ne_param.num_U ne_param.num_K]);
for i_u = 1 : ne_param.num_U
    u = ne_param.U(i_u);
    for i_k = 1 : ne_param.num_K
        m = m_down_u_k(i_u,i_k);
        for i_uj = 1 : ne_param.num_U
            for i_kj = 1 : ne_param.num_K
                mj = ne_m_down_u_k(i_uj,i_kj);
                delta_down_u_k_uj_kj(i_u,i_k,i_uj,i_kj) = ...
                    piecewise(m < mj, u, m == mj, 0.5 * u, 0);
            end
        end
    end
end

%% lambda(m) tensor
lambda_down_u_k_uj_kj_up_un_kn = sym('lambda', [ne_param.num_U ne_param.num_K ne_param.num_U ne_param.num_K ne_param.num_U ne_param.num_K]);
for i_u = 1 : ne_param.num_U
    for i_k = 1 : ne_param.num_K
        k = ne_param.K(i_k);
        m = m_down_u_k(i_u,i_k);
        z = floor(m);
        r = m - z;
        for i_uj = 1 : ne_param.num_U
            for i_kj = 1 : ne_param.num_K
                kj = ne_param.num_K;
                mj = ne_m_down_u_k(i_uj,i_kj);
                zj = floor(mj);
                rj = mj - zj;
                min_z_k_max_kj = piecewise(z < ne_param.k_max - kj, z, ne_param.k_max - kj);
                min_z_1_k_max_kj = piecewise(z + 1 < ne_param.k_max - kj, z + 1, ne_param.k_max - kj);
                min_k_zj_k_max = min([k + zj, ne_param.k_max]);
                min_k_zj_1_k_max = min([k + zj + 1, ne_param.k_max]);
                for i_un = 1 : ne_param.num_U
                    mu = ne_param.mu_down_u_up_un(i_u,i_un);
                    for i_kn = 1 : ne_param.num_K
                        fprintf('i_u %d i_k %d i_uj %d i_kj %d i_un %d i_kn %d\n', i_u, i_k, i_uj, i_kj, i_un, i_kn);
                        kn = ne_param.K(i_kn);
                        lambda_down_u_k_uj_kj_up_un_kn(i_u,i_k,i_uj,i_kj,i_un,i_kn) = ...
                            mu * (piecewise(m > mj & kn == k - min_z_k_max_kj, 1 - r, 0)...
                            + piecewise(m > mj & kn == k - min_z_1_k_max_kj, r, 0)...
                            + piecewise(m < mj & kn == min_k_zj_k_max, 1 - rj, 0)...
                            + piecewise(m < mj & kn == min_k_zj_1_k_max, rj, 0)...
                            + piecewise(m == mj & kn == k - min_z_k_max_kj, 0.5 * (1 - rj), 0)...
                            + piecewise(m == mj & kn == k - min_z_1_k_max_kj, 0.5 * rj, 0)...
                            + piecewise(m == mj & kn == min_k_zj_k_max, 0.5 * (1 - rj), 0)...
                            + piecewise(m == mj & kn == min_k_zj_1_k_max, 0.5 * rj, 0));
                    end
                end
            end
        end
    end
end