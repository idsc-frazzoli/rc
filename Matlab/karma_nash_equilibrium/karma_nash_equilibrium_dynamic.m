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

%% Start with a NE guess policy function
% Functions take the following form:
% pi_down_0(k) = 0 (bid 0 if non-urgent)
% pi_down_U(k) = round(a * k + b * log(k + 1)) (combination of linear and
% logarithmic term if urgent)
% Note that policy functions are pure. They are characterized by parameters
% a & b
a = 0.5;
% b = (0.5 - a) / log(2);
% b = 1 - a;
% b = ((0.5 - a) / log(2) + 1 - a) / 2;
b = 3;
c = 2;

%% Guess an initial distribution of urgency and karma
% Set initial distribution of urgency to stationary distribution of urgency
% markov chain 
p_up_u = ne_func.stat_dist(ne_param.mu_down_u_up_un);

% Set initial distribution of karma to be all at k_ave
k_max = ne_param.k_ave;
K = 0 : k_max;
num_K = length(K);
s_up_k = zeros(num_K, 1);
s_up_k(end) = 1;
% k_max = ne_param.k_ave * 2;
% K = 0 : k_max;
% num_K = length(K);
% s_up_k = 1 / num_K * ones(num_K, 1);

%% Get probability distribution over seeing messages
% M_k = round(a * K + b * log(K + 1));
M_k_lin = round(a * K(K <= b));
M_k_log = round(a * b + c * log(K(K > b) - b + 1));
M_k = [M_k_lin, M_k_log];
[M, i_M_k, ] = unique(M_k);
num_M = length(M);
iota_up_m = zeros(num_M, 1);
iota_up_m(1) = p_up_u(1) + p_up_u(end) * s_up_k(1);
for i_m = 2 : num_M - 1
    iota_up_m(i_m) = p_up_u(end) * sum(s_up_k(i_M_k(i_m):i_M_k(i_m+1)-1));
end
iota_up_m(end) = p_up_u(end) * sum(s_up_k(i_M_k(end):end));

%% Get karma transition matrices for non-urgent and urgent agents
% Matrices are 'fat'. We consider only next karma values that are possible
kn_max = k_max + M(end);
num_Kn = kn_max + 1;
t_down_0_k_up_kn = zeros(num_K, num_Kn);
t_down_U_k_up_kn = zeros(num_K, num_Kn);
t_down_0_k_up_kn(1,1+M) = iota_up_m.';
t_down_U_k_up_kn(1,1+M) = iota_up_m.';
for i_k = 2 : num_K
    t_down_0_k_up_kn(i_k,i_k+M) = iota_up_m.';
    m = M_k(i_k);
    i_m_win = find(M < m);
    i_kn_win = i_k - m;
    i_m_tie = find(M == m);
    i_kn_tie = i_k + m;
    i_m_lose = find(M > m);
    i_kn_lose = i_k + M(i_m_lose);
    t_down_U_k_up_kn(i_k,i_kn_win) = sum(iota_up_m(i_m_win)) + 0.5 * iota_up_m(i_m_tie);
    t_down_U_k_up_kn(i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
    t_down_U_k_up_kn(i_k,i_kn_lose) = iota_up_m(i_m_lose).';
end
t_down_k_up_kn = p_up_u(1) * t_down_0_k_up_kn + p_up_u(end) * t_down_U_k_up_kn;

%% Now take a step in the direction of transition matrix
s_up_k_next = t_down_k_up_kn.' * s_up_k;

%% Now pluck out unlikely states form the end
while s_up_k_next(end) < 1e-8
    s_up_k_next(end) = [];
end

%% Renormalize
s_up_k_next = s_up_k_next / sum(s_up_k_next);

%% Repeat all of the above until convegrence
if length(s_up_k) == length(s_up_k_next)
    converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
else
    converged = false;
end
s_up_k = s_up_k_next;
while ~converged
    k_max = length(s_up_k) - 1;
    K = 0 : k_max;
    num_K = length(K);

    %% Get probability distribution over seeing messages
%     M_k = round(a * K + b * log(K + 1));
    M_k_lin = round(a * K(K <= b));
    M_k_log = round(a * b + c * log(K(K > b) - b + 1));
    M_k = [M_k_lin, M_k_log];
    [M, i_M_k, ] = unique(M_k);
    num_M = length(M);
    iota_up_m = zeros(num_M, 1);
    iota_up_m(1) = p_up_u(1) + p_up_u(end) * s_up_k(1);
    for i_m = 2 : num_M - 1
        iota_up_m(i_m) = p_up_u(end) * sum(s_up_k(i_M_k(i_m):i_M_k(i_m+1)-1));
    end
    iota_up_m(end) = p_up_u(end) * sum(s_up_k(i_M_k(end):end));

    %% Get karma transition matrices for non-urgent and urgent agents
    % Matrices are 'fat'. We consider only next karma values that are possible
    kn_max = k_max + M(end);
    num_Kn = kn_max + 1;
    t_down_0_k_up_kn = zeros(num_K, num_Kn);
    t_down_U_k_up_kn = zeros(num_K, num_Kn);
    t_down_0_k_up_kn(1,1+M) = iota_up_m.';
    t_down_U_k_up_kn(1,1+M) = iota_up_m.';
    for i_k = 2 : num_K
        t_down_0_k_up_kn(i_k,i_k+M) = iota_up_m.';
        m = M_k(i_k);
        i_m_win = find(M < m);
        i_kn_win = i_k - m;
        i_m_tie = find(M == m);
        i_kn_tie = i_k + m;
        i_m_lose = find(M > m);
        i_kn_lose = i_k + M(i_m_lose);
        t_down_U_k_up_kn(i_k,i_kn_win) = sum(iota_up_m(i_m_win)) + 0.5 * iota_up_m(i_m_tie);
        t_down_U_k_up_kn(i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
        t_down_U_k_up_kn(i_k,i_kn_lose) = iota_up_m(i_m_lose).';
    end
    t_down_k_up_kn = p_up_u(1) * t_down_0_k_up_kn + p_up_u(end) * t_down_U_k_up_kn;

    %% Now take a step in the direction of transition matrix
    s_up_k_next = t_down_k_up_kn.' * s_up_k;

    %% Now pluck out unlikely states form the end
    while s_up_k_next(end) < 1e-8
        s_up_k_next(end) = [];
    end

    %% Renormalize
    s_up_k_next = s_up_k_next / sum(s_up_k_next);

    %% Repeat all of the above until convegrence
    if length(s_up_k) == length(s_up_k_next)
        converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
    else
        converged = false;
    end
    s_up_k = s_up_k_next;
end

%% Get expected stage cost
q_down_0_k = zeros(num_K, 1);
q_down_U_k = zeros(num_K, 1);
for i_k = 1 : num_K
    m = M_k(i_k);
    i_m_tie = find(M == m);
    i_m_lose = find(M > m);
    q_down_U_k(i_k) = (0.5 * iota_up_m(i_m_tie) + sum(iota_up_m(i_m_lose))) * ne_param.U(end);
end

%% Concatenate
q_down_u_k = [q_down_0_k; q_down_U_k];
% Get rid of 'extra' states. Is this the same as saturation?
t_down_0_k_up_kn_trunc = t_down_0_k_up_kn(:,1:num_K);
t_down_0_k_up_kn_trunc(:,end) = 1 - sum(t_down_0_k_up_kn_trunc(:,1:end-1), 2);
t_down_U_k_up_kn_trunc = t_down_U_k_up_kn(:,1:num_K);
t_down_U_k_up_kn_trunc(:,end) = 1 - sum(t_down_U_k_up_kn_trunc(:,1:end-1), 2);
t_down_u_k_up_un_kn = [ne_param.mu_down_u_up_un(1,1) * t_down_0_k_up_kn_trunc, ne_param.mu_down_u_up_un(1,2) * t_down_0_k_up_kn_trunc;...
    ne_param.mu_down_u_up_un(2,1) * t_down_U_k_up_kn_trunc, ne_param.mu_down_u_up_un(2,2) * t_down_U_k_up_kn_trunc];

%% Set up optimization to find alpha for which guessed policy is NE
alpha = sdpvar(1, 1, 'full');
v_down_u_k = sdpvar(ne_param.num_U * num_K, 1, 'full');

constraints = alpha >= 0;
constraints = [constraints; alpha <= 1];
constraints = [constraints; v_down_u_k <= q_down_u_k + alpha * t_down_u_k_up_un_kn * v_down_u_k];
constraints = [constraints; v_down_u_k >= q_down_u_k + alpha * t_down_u_k_up_un_kn * v_down_u_k];
% Only optimize over 'prevalent' k's
s_up_k_trunc = s_up_k;
while s_up_k_trunc(end) < 1e-3
    s_up_k_trunc(end) = [];
end
for i_k = 3 : length(s_up_k_trunc)
    for m = 1 : K(i_k)
        i_m_win = find(M < m);
        i_m_tie = find(M == m);
        i_m_lose = find(M > m);
        i_kn_win = i_k - m;
        i_kn_tie = i_k + m;
        i_kn_lose = i_k + M(i_m_lose);
        q_down_U_k_m = 0;
        t_down_U_k_m_up_kn = zeros(1, num_Kn);
        if ~isempty(i_m_win)
            t_down_U_k_m_up_kn(i_kn_win) = sum(iota_up_m(i_m_win));
        end
        if ~isempty(i_m_tie)
            q_down_U_k_m = 0.5 * iota_up_m(i_m_tie) * ne_param.U(end);
            t_down_U_k_m_up_kn(i_kn_win) = t_down_U_k_m_up_kn(i_kn_win) + 0.5 * iota_up_m(i_m_tie);
            t_down_U_k_m_up_kn(i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
        end
        if ~isempty(i_m_lose)
            q_down_U_k_m = q_down_U_k_m + sum(iota_up_m(i_m_lose)) * ne_param.U(end);
            t_down_U_k_m_up_kn(i_kn_lose) = iota_up_m(i_m_lose).';
        end
        t_down_U_k_m_up_kn_trunc = t_down_U_k_m_up_kn(1:num_K);
        t_down_U_k_m_up_kn_trunc(end) = 1 - sum(t_down_U_k_m_up_kn_trunc(1:end-1));
        t_down_U_k_m_up_un_kn = [ne_param.mu_down_u_up_un(2,1) * t_down_U_k_m_up_kn_trunc, ne_param.mu_down_u_up_un(2,2) * t_down_U_k_m_up_kn_trunc];
        constraints = [constraints; v_down_u_k(num_K+i_k) <= q_down_U_k_m + alpha * t_down_U_k_m_up_un_kn * v_down_u_k];
    end
end

objective = alpha;
% options = sdpsettings('solver', 'fmincon', 'fmincon.MaxIter', 10000, 'fmincon.MaxFunEvals', 10000, 'verbose', 3);
options = sdpsettings('solver', 'knitro', 'verbose', 3);
diagnostics_min = optimize(constraints, objective, options)

alpha_min = double(alpha);

objective = -alpha;
diagnostics_max = optimize(constraints, objective, options)

alpha_max = double(alpha);

%% Inform user when done
fprintf('DONE\n\n');