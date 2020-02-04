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

%% Guess an initial distribution of urgency and karma
% Set initial distribution of urgency to stationary distribution of urgency
% markov chain 
p_up_u = ne_func.stat_dist(ne_param.mu_down_u_up_un);

% Set initial distribution of karma to be all at k_ave
% k_max = ne_param.k_ave;
% K = 0 : k_max;
% num_K = length(K);
% s_up_k = zeros(num_K, 1);
% s_up_k(end) = 1;
% Set initial distribution of karma to be uniform around k_ave
k_max = ne_param.k_ave * 2;
K = (0 : k_max).';
num_K = length(K);
s_up_k = 1 / num_K * ones(num_K, 1);

%% Guess policy for initial set of K
% Policy will be extended to last value guessed if K grows
m_down_U_k = round(0.5 * K);

%% Get probability distribution over seeing messages
[M, i_m_down_U_k] = unique(m_down_U_k);
num_M = length(M);
iota_up_m = zeros(num_M, 1);
iota_up_m(1) = p_up_u(1) + p_up_u(end) * s_up_k(1);
for i_m = 2 : num_M - 1
    iota_up_m(i_m) = p_up_u(end) * sum(s_up_k(i_m_down_U_k(i_m):i_m_down_U_k(i_m+1)-1));
end
iota_up_m(end) = p_up_u(end) * sum(s_up_k(i_m_down_U_k(end):end));

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
    m = m_down_U_k(i_k);
    i_m_win = find(M < m);
    i_kn_win = i_k - m;
    t_down_U_k_up_kn(i_k,i_kn_win) = sum(iota_up_m(i_m_win));
    i_m_tie = find(M == m);
    if ~isempty(i_m_tie)
        i_kn_tie = i_k + m;
        t_down_U_k_up_kn(i_k,i_kn_win) = t_down_U_k_up_kn(i_k,i_kn_win) + 0.5 * iota_up_m(i_m_tie);
        t_down_U_k_up_kn(i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
    end
    i_m_lose = find(M > m);
    if ~isempty(i_m_lose)
        i_kn_lose = i_k + M(i_m_lose);
        t_down_U_k_up_kn(i_k,i_kn_lose) = iota_up_m(i_m_lose).';
    end
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
    s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
else
    s_converged = false;
end
s_up_k = s_up_k_next;
while ~s_converged
    k_max = length(s_up_k) - 1;
    K = 0 : k_max;
    num_K = length(K);

    %% Get probability distribution over seeing messages
    if num_K > length(m_down_U_k)
        m_down_U_k = [m_down_U_k; repmat(m_down_U_k(end), num_K - length(m_down_U_k), 1)];
    elseif num_K < length(m_down_U_k)
        m_down_U_k(num_K+1:end) = [];
    end
    [M, i_m_down_U_k] = unique(m_down_U_k);
    num_M = length(M);
    iota_up_m = zeros(num_M, 1);
    iota_up_m(1) = p_up_u(1) + p_up_u(end) * s_up_k(1);
    for i_m = 2 : num_M - 1
        iota_up_m(i_m) = p_up_u(end) * sum(s_up_k(i_m_down_U_k(i_m):i_m_down_U_k(i_m+1)-1));
    end
    iota_up_m(end) = p_up_u(end) * sum(s_up_k(i_m_down_U_k(end):end));

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
        m = m_down_U_k(i_k);
        i_m_win = find(M < m);
        i_kn_win = i_k - m;
        t_down_U_k_up_kn(i_k,i_kn_win) = sum(iota_up_m(i_m_win));
        i_m_tie = find(M == m);
        if ~isempty(i_m_tie)
            i_kn_tie = i_k + m;
            t_down_U_k_up_kn(i_k,i_kn_win) = t_down_U_k_up_kn(i_k,i_kn_win) + 0.5 * iota_up_m(i_m_tie);
            t_down_U_k_up_kn(i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
        end
        i_m_lose = find(M > m);
        if ~isempty(i_m_lose)
            i_kn_lose = i_k + M(i_m_lose);
            t_down_U_k_up_kn(i_k,i_kn_lose) = iota_up_m(i_m_lose).';
        end
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
        s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
    else
        s_converged = false;
    end
    s_up_k = s_up_k_next;
end

%% Get expected stage cost
q_down_0_k = zeros(num_K, 1);
q_down_U_k = zeros(num_K, 1);
for i_k = 1 : num_K
    m = m_down_U_k(i_k);
    i_m_tie = find(M == m);
    if ~isempty(i_m_tie)
        q_down_U_k(i_k) = 0.5 * iota_up_m(i_m_tie) * ne_param.U(end);
    end
    i_m_lose = find(M > m);
    q_down_U_k(i_k) = q_down_U_k(i_k) + sum(iota_up_m(i_m_lose)) * ne_param.U(end);
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

%% Get expected infinite horizon cost
v_down_u_k = zeros(ne_param.num_U * num_K, 1);
v_down_u_k_next = q_down_u_k + ne_param.alpha * t_down_u_k_up_un_kn * v_down_u_k;
v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
v_down_u_k = v_down_u_k_next;
while ~v_converged
    v_down_u_k_next = q_down_u_k + ne_param.alpha * t_down_u_k_up_un_kn * v_down_u_k;
    v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
    v_down_u_k = v_down_u_k_next;
end

%% Get best response
% Only do so for 'prevalent' k's
s_up_k_trunc = s_up_k;
% while s_up_k_trunc(end) < 1e-3
%     s_up_k_trunc(end) = [];
% end
num_K_trunc = length(s_up_k_trunc);
br_m_down_U_k = m_down_U_k(1:num_K_trunc);
for i_k = 3 : num_K_trunc
    v_down_U_k_min = v_down_u_k(num_K + i_k);
    for m = br_m_down_U_k(i_k-1) : K(i_k)
        if m == m_down_U_k(i_k)
            continue;
        end
        q_down_U_k_m = 0;
        t_down_U_k_m_up_kn = zeros(1, num_Kn);
        i_m_win = find(M < m);
        i_kn_win = i_k - m;
        t_down_U_k_m_up_kn(i_kn_win) = sum(iota_up_m(i_m_win));
        i_m_tie = find(M == m);
        if ~isempty(i_m_tie)
            i_kn_tie = i_k + m;
            q_down_U_k_m = 0.5 * iota_up_m(i_m_tie) * ne_param.U(end);
            t_down_U_k_m_up_kn(i_kn_win) = t_down_U_k_m_up_kn(i_kn_win) + 0.5 * iota_up_m(i_m_tie);
            t_down_U_k_m_up_kn(i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
        end
        i_m_lose = find(M > m);
        if ~isempty(i_m_lose)
            i_kn_lose = i_k + M(i_m_lose);
            q_down_U_k_m = q_down_U_k_m + sum(iota_up_m(i_m_lose)) * ne_param.U(end);
            t_down_U_k_m_up_kn(i_kn_lose) = iota_up_m(i_m_lose).';
        end
        t_down_U_k_m_up_kn_trunc = t_down_U_k_m_up_kn(1:num_K);
        t_down_U_k_m_up_kn_trunc(end) = 1 - sum(t_down_U_k_m_up_kn_trunc(1:end-1));
        t_down_U_k_m_up_un_kn = [ne_param.mu_down_u_up_un(2,1) * t_down_U_k_m_up_kn_trunc, ne_param.mu_down_u_up_un(2,2) * t_down_U_k_m_up_kn_trunc];
        v_down_U_k_m = q_down_U_k_m + ne_param.alpha * t_down_U_k_m_up_un_kn * v_down_u_k;
        if v_down_U_k_m < v_down_U_k_min - ne_param.br_v_tol
            br_m_down_U_k(i_k) = m;
            v_down_U_k_min = v_down_U_k_m;
        end
    end
    if br_m_down_U_k(i_k) < br_m_down_U_k(i_k-1)
        br_m_down_U_k(i_k) = br_m_down_U_k(i_k-1);
    end
end

%% Repeat until convergence
ne_converged = norm(m_down_U_k(1:num_K_trunc) - br_m_down_U_k, inf) <= ne_param.br_pi_tol;
m_down_U_k = [br_m_down_U_k; repmat(br_m_down_U_k(end), num_K - num_K_trunc, 1)];
num_ne_iter = 1;
% Display status and store history of policies
for i_k = 1 : num_K
    if i_k == 1
        fprintf('Iteration %d policy: %d', num_ne_iter, m_down_U_k(i_k));
    else
        fprintf('->%d', m_down_U_k(i_k));
    end
end
ne_m_down_U_k_hist = m_down_U_k.';
ne_num_K_hist = size(ne_m_down_U_k_hist, 2);
fprintf('\n\n');
while ~ne_converged
    %% Get probability distribution over seeing messages
    [M, i_m_down_U_k, ] = unique(m_down_U_k);
    num_M = length(M);
    iota_up_m = zeros(num_M, 1);
    iota_up_m(1) = p_up_u(1) + p_up_u(end) * s_up_k(1);
    for i_m = 2 : num_M - 1
        iota_up_m(i_m) = p_up_u(end) * sum(s_up_k(i_m_down_U_k(i_m):i_m_down_U_k(i_m+1)-1));
    end
    iota_up_m(end) = p_up_u(end) * sum(s_up_k(i_m_down_U_k(end):end));

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
        m = m_down_U_k(i_k);
        i_m_win = find(M < m);
        i_kn_win = i_k - m;
        t_down_U_k_up_kn(i_k,i_kn_win) = sum(iota_up_m(i_m_win));
        i_m_tie = find(M == m);
        if ~isempty(i_m_tie)
            i_kn_tie = i_k + m;
            t_down_U_k_up_kn(i_k,i_kn_win) = t_down_U_k_up_kn(i_k,i_kn_win) + 0.5 * iota_up_m(i_m_tie);
            t_down_U_k_up_kn(i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
        end
        i_m_lose = find(M > m);
        if ~isempty(i_m_lose)
            i_kn_lose = i_k + M(i_m_lose);
            t_down_U_k_up_kn(i_k,i_kn_lose) = iota_up_m(i_m_lose).';
        end
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
        s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
    else
        s_converged = false;
    end
    s_up_k = s_up_k_next;
    while ~s_converged
        k_max = length(s_up_k) - 1;
        K = 0 : k_max;
        num_K = length(K);

        %% Get probability distribution over seeing messages
        if num_K > length(m_down_U_k)
            m_down_U_k = [m_down_U_k; repmat(m_down_U_k(end), num_K - length(m_down_U_k), 1)];
        elseif num_K < length(m_down_U_k)
            m_down_U_k(num_K+1:end) = [];
        end
        [M, i_m_down_U_k, ] = unique(m_down_U_k);
        num_M = length(M);
        iota_up_m = zeros(num_M, 1);
        iota_up_m(1) = p_up_u(1) + p_up_u(end) * s_up_k(1);
        for i_m = 2 : num_M - 1
            iota_up_m(i_m) = p_up_u(end) * sum(s_up_k(i_m_down_U_k(i_m):i_m_down_U_k(i_m+1)-1));
        end
        iota_up_m(end) = p_up_u(end) * sum(s_up_k(i_m_down_U_k(end):end));

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
            m = m_down_U_k(i_k);
            i_m_win = find(M < m);
            i_kn_win = i_k - m;
            t_down_U_k_up_kn(i_k,i_kn_win) = sum(iota_up_m(i_m_win));
            i_m_tie = find(M == m);
            if ~isempty(i_m_tie)
                i_kn_tie = i_k + m;
                t_down_U_k_up_kn(i_k,i_kn_win) = t_down_U_k_up_kn(i_k,i_kn_win) + 0.5 * iota_up_m(i_m_tie);
                t_down_U_k_up_kn(i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
            end
            i_m_lose = find(M > m);
            if ~isempty(i_m_lose)
                i_kn_lose = i_k + M(i_m_lose);
                t_down_U_k_up_kn(i_k,i_kn_lose) = iota_up_m(i_m_lose).';
            end
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
            s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
        else
            s_converged = false;
        end
        s_up_k = s_up_k_next;
    end
    
    %% Get expected stage cost
    q_down_0_k = zeros(num_K, 1);
    q_down_U_k = zeros(num_K, 1);
    for i_k = 1 : num_K
        m = m_down_U_k(i_k);
        i_m_tie = find(M == m);
        if ~isempty(i_m_tie)
            q_down_U_k(i_k) = 0.5 * iota_up_m(i_m_tie) * ne_param.U(end);
        end
        i_m_lose = find(M > m);
        q_down_U_k(i_k) = q_down_U_k(i_k) + sum(iota_up_m(i_m_lose)) * ne_param.U(end);
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

    %% Get expected infinite horizon cost
    v_down_u_k = zeros(ne_param.num_U * num_K, 1);
    v_down_u_k_next = q_down_u_k + ne_param.alpha * t_down_u_k_up_un_kn * v_down_u_k;
    v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
    v_down_u_k = v_down_u_k_next;
    while ~v_converged
        v_down_u_k_next = q_down_u_k + ne_param.alpha * t_down_u_k_up_un_kn * v_down_u_k;
        v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
        v_down_u_k = v_down_u_k_next;
    end

    %% Get best response
    % Only do so for 'prevalent' k's
    s_up_k_trunc = s_up_k;
%     while s_up_k_trunc(end) < 1e-3
%         s_up_k_trunc(end) = [];
%     end
    num_K_trunc = length(s_up_k_trunc);
    br_m_down_U_k = m_down_U_k(1:num_K_trunc);
    for i_k = 3 : num_K_trunc
        v_down_U_k_min = v_down_u_k(num_K + i_k);
        for m = br_m_down_U_k(i_k-1) : K(i_k)
            if m == m_down_U_k(i_k)
                continue;
            end
            q_down_U_k_m = 0;
            t_down_U_k_m_up_kn = zeros(1, num_Kn);
            i_m_win = find(M < m);
            i_kn_win = i_k - m;
            t_down_U_k_m_up_kn(i_kn_win) = sum(iota_up_m(i_m_win));
            i_m_tie = find(M == m);
            if ~isempty(i_m_tie)
                i_kn_tie = i_k + m;
                q_down_U_k_m = 0.5 * iota_up_m(i_m_tie) * ne_param.U(end);
                t_down_U_k_m_up_kn(i_kn_win) = t_down_U_k_m_up_kn(i_kn_win) + 0.5 * iota_up_m(i_m_tie);
                t_down_U_k_m_up_kn(i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
            end
            i_m_lose = find(M > m);
            if ~isempty(i_m_lose)
                i_kn_lose = i_k + M(i_m_lose);
                q_down_U_k_m = q_down_U_k_m + sum(iota_up_m(i_m_lose)) * ne_param.U(end);
                t_down_U_k_m_up_kn(i_kn_lose) = iota_up_m(i_m_lose).';
            end
            t_down_U_k_m_up_kn_trunc = t_down_U_k_m_up_kn(1:num_K);
            t_down_U_k_m_up_kn_trunc(end) = 1 - sum(t_down_U_k_m_up_kn_trunc(1:end-1));
            t_down_U_k_m_up_un_kn = [ne_param.mu_down_u_up_un(2,1) * t_down_U_k_m_up_kn_trunc, ne_param.mu_down_u_up_un(2,2) * t_down_U_k_m_up_kn_trunc];
            v_down_U_k_m = q_down_U_k_m + ne_param.alpha * t_down_U_k_m_up_un_kn * v_down_u_k;
            if v_down_U_k_m < v_down_U_k_min - ne_param.br_v_tol
                br_m_down_U_k(i_k) = m;
                v_down_U_k_min = v_down_U_k_m;
            end
        end
        if br_m_down_U_k(i_k) < br_m_down_U_k(i_k-1)
            br_m_down_U_k(i_k) = br_m_down_U_k(i_k-1);
        end
    end

    ne_converged = norm(m_down_U_k(1:num_K_trunc) - br_m_down_U_k, inf) <= ne_param.br_pi_tol;
    m_down_U_k = [br_m_down_U_k; repmat(br_m_down_U_k(end), num_K - num_K_trunc, 1)];
    num_ne_iter = num_ne_iter + 1;
    % Display status and store history of policies
    for i_k = 1 : num_K
        if i_k == 1
            fprintf('Iteration %d policy: %d', num_ne_iter, m_down_U_k(i_k));
        else
            fprintf('->%d', m_down_U_k(i_k));
        end
    end
    
    % Process history
    if num_K > ne_num_K_hist
        ne_m_down_U_k_hist = [ne_m_down_U_k_hist, repmat(ne_m_down_U_k_hist(:,end), 1, num_K - ne_num_K_hist)];
        ne_m_down_U_k_hist_end = m_down_U_k.';
    elseif num_K < ne_num_K_hist
        ne_m_down_U_k_hist_end = [m_down_U_k.', repmat(m_down_U_k(end), 1, ne_num_K_hist - num_K)];
    else
        ne_m_down_U_k_hist_end = m_down_U_k.';
    end
    ne_num_K_hist = size(ne_m_down_U_k_hist, 2);
    
    % Detect a limit cycle
    limit_cycle = false;
    for ne_hist_i = 1 : size(ne_m_down_U_k_hist, 1)
        if isequal(ne_m_down_U_k_hist(ne_hist_i,:), ne_m_down_U_k_hist_end)
            % Limit cycle found
            ne_m_down_U_k_limit_cycle = ne_m_down_U_k_hist(ne_hist_i:end,:);
            ne_limit_cycle_code = ne_m_down_U_k_limit_cycle * (1 : ne_num_K_hist).';
            if size(ne_m_down_U_k_limit_cycle, 1) > 1
                limit_cycle = true;
            end
            break;
        end
    end
    ne_m_down_U_k_hist = [ne_m_down_U_k_hist; ne_m_down_U_k_hist_end];
    fprintf('\n\n');
    if  limit_cycle
        fprintf('Limit cycle found!\n\n');
        break;
    end
end

%% Result
if  limit_cycle
    ne_m_down_U_k = mean(ne_m_down_U_k_limit_cycle).';
else
    ne_m_down_U_k = m_down_U_k;
end

%% Inform user when done
fprintf('DONE\n\n');