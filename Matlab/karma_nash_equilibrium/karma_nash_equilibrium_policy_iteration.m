clear;
close all;
clc;

%% Parameters
param = load_parameters();

%% Constants
% Vector of all karma values
const.k = (param.k_min : 1 : param.k_max).';
const.num_k = length(const.k);

% k_next cell of matrices. Each pair of 2 agents' (k_i, k_j) has a matrix
% (i,j). Each matrix describes next karma for agent i as a function of
% their bid m_i (rows) and other agent j's bid m_j (cols). Note that agents
% are limited in their bids by their karma level, which is why matrices
% have different dimensions
const.k_next = cell(const.num_k);
for i_k_i = 1 : const.num_k
    k_i = const.k(i_k_i);
    for i_k_j = 1 : const.num_k
        k_j = const.k(i_k_j);
        const.k_next{i_k_i,i_k_j} = cell(i_k_i, i_k_j);
        for i_m_i = 1 : i_k_i
            m_i = const.k(i_m_i);
            for i_m_j = 1 : i_k_j
                m_j = const.k(i_m_j);
                
                % Next karma level if agent i is to receive karma
                k_in = min([k_i + m_j, param.k_max]);
                % Next karma level if agent i is to pay karma
                k_out = k_i - min([m_i, param.k_max - k_j]);
                
                % Agent i receives karma when they bid lower than agent j
                if m_i < m_j
                    const.k_next{i_k_i,i_k_j}{i_m_i,i_m_j} = k_in;
                % Agent i pays karma when they bid higher than agent j
                elseif m_i > m_j
                    const.k_next{i_k_i,i_k_j}{i_m_i,i_m_j} = k_out;
                % Agent i can either pay or receive karma on equal bids
                % (50/50 chances). We keep track of both options here
                else
                    const.k_next{i_k_i,i_k_j}{i_m_i,i_m_j} = [k_in, k_out];
                end
            end
        end
    end
end

% Future discount factor
const.alpha = 0.8;

% Tolerance for convergence
const.tol = 1e-6;

% Maximum number of iterations
const.max_iter = 1000;

%% Iterative algorithm to find Karma Nash equilibrium %%

%% Step 0: Initialization
% Step 0.1: NE policy matrix for non-zero urgency guess
% Parametrized as a cell matrix. Row i in cell corresponds to policy for
% karma level k(i) and has only i columns since agents cannot bid more than
% their karma. Column j in row i denotes probability of transmitting
% message k(j) when karma level is k(i)
% Note 1: All entries are non-negative and rows sum to 1
% Note 2: NE policy for urgency 0 is to bid 0 karma always. This is
% hardcoded in algorithm and not parametrized
% Initialize uniformly at random
policy = cell(const.num_k, 1);
for i_k_i = 1 : const.num_k
    policy{i_k_i} = rand(1, i_k_i);
    policy{i_k_i} = policy{i_k_i} / sum(policy{i_k_i});
end
% for i_k_i = 1 : const.num_k
%     policy{i_k_i} = zeros(1, i_k_i);
%     policy{i_k_i}(i_k_i) = 1;
% end
% for i_k_i = 1 : const.num_k
%     policy{i_k_i} = zeros(1, i_k_i);
% end
% policy{1}(1) = 1;
% policy{2}(2) = 1;
% policy{3}(2) = 1;
% policy{4}(2) = 1;
% policy{5}(3) = 1;
% policy{6}(3) = 1;
% policy{7}(3) = 1;
% policy{8}(4) = 1;
% policy{9}(4) = 1;
% policy{10}(4) = 1;
% policy{11}(5) = 1;
% policy{12}(5) = 1;
% policy{13}(5) = 1;

% Step 0.2: Stationary distribution at NE policy guess
% Parametrized as a vector with num_k rows. D(i) is probability of k = k(i)
% Note: All entries are non-negative and vector sums to 1
% Initialize uniformly at random
% D = rand(const.num_k, 1);
% D = D / sum(D);
D = 1 / const.num_k * ones(const.num_k, 1);
% D = zeros(const.num_k, 1);
% D(1) = 0.5;
% D(end) = 0.5;

%% Step 1: Get (D,T) pair corresponding to NE policy guess
% Step 1.1: Get T for when all agents play policy and stationary
% distribution is D
T = ne_func.get_T(policy, policy, D, const, param);

% Step 1.2: Get stattionary distribution D from T
D_next = ne_func.get_D(T, const);

% Step 1.3: Repeat steps 1.1-1.2 until (D,T) pair converge
num_iter_D_T = 0;
while norm(D - D_next, inf) > const.tol && num_iter_D_T < const.max_iter
    D = D_next;
    T = ne_func.get_T(policy, policy, D, const, param);
    D_next = ne_func.get_D(T, const);
    num_iter_D_T = num_iter_D_T + 1;
end
if num_iter_D_T == const.max_iter
    fprintf('Iteration %d did not find convergent (D,T) pair\n', num_iter);
end

%% Step 2: Best response of agent i to all other agents j playing policy
% This is a dynamic progarmming problem solved using policy iteration
% Step 2.0: Initialize agent i's best response policy with current NE guess
policy_i = policy;
% Step 2.1: Get current stage cost for agent i playing policy_i
c_i = ne_func.get_c(policy_i, policy, D, const, param);
% Step 2.2: Get probability transition matrix for agent i playing policy_i
T_i = ne_func.get_T(policy_i, policy, D, const, param);
% Step 2.3: Get expected utility for agent i playing policy_i
theta_i = (eye(const.num_k) - const.alpha * T_i) \ c_i;
% Step 2.4: Get the expected cost matrix for agent i as per the messages
% they would transmit, given current expected utility theta_i
rho_i = ne_func.get_rho(policy, D, theta_i, const, param);
% Step 2.5: Get next minimizing policy for agent i
policy_i_next = ne_func.get_policy(rho_i, const);
% Step 2.6: Policy iteration - repeat Steps 2.1-2.5 until policy_i
% converges
policy_i_error = ne_func.policy_norm(policy_i, policy_i_next, inf);
num_iter = 0;
num_policy_iter = 0;
% Display status
fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_iter, num_policy_iter, policy_i_error);
while policy_i_error > const.tol && num_policy_iter < const.max_iter
    num_policy_iter = num_policy_iter + 1;
    
    policy_i = policy_i_next;
    c_i = ne_func.get_c(policy_i, policy, D, const, param);
    T_i = ne_func.get_T(policy_i, policy, D, const, param);
    theta_i = (eye(const.num_k) - const.alpha * T_i) \ c_i;
    rho_i = ne_func.get_rho(policy, D, theta_i, const, param);
    policy_i_next = ne_func.get_policy(rho_i, const);
    policy_i_error = ne_func.policy_norm(policy_i, policy_i_next, inf);
    
    % Display status
    fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_iter, num_policy_iter, policy_i_error);
end
policy_next = policy_i;

%% Step 3: Repeat Step 1-2 until convergence
policy_error = ne_func.policy_norm(policy, policy_next, inf);
% Display status
fprintf('Iteration %d policy error %f\n', num_iter, policy_error);
policy_hist_end = zeros(1, const.num_k);
policy_hist_end(1) = const.k(policy_next{1}==1);
fprintf('Iteration %d policy: %d', num_iter, policy_hist_end(1));
for i_k_i = 2 : const.num_k
    policy_hist_end(i_k_i) = const.k(policy_next{i_k_i}==1);
    fprintf("->%d", policy_hist_end(i_k_i));
end
policy_hist = policy_hist_end;
fprintf('\n\n');
while policy_error > const.tol && num_iter < const.max_iter
    num_iter = num_iter + 1;
    policy = policy_next;
    
    % Step 3-1.1: Get T for when all agents play policy and stationary
    % distribution is D
    % Re-initialize D to uniform distribution first
    D = 1 / const.num_k * ones(const.num_k, 1);
    T = ne_func.get_T(policy, policy, D, const, param);

    % Step 3-1.2: Get stattionary distribution D from T
    D_next = ne_func.get_D(T, const);

    % Step 3-1.3: Repeat steps 1.1-1.2 until (D,T) pair converge
    num_iter_D_T = 0;
    while norm(D - D_next, inf) > const.tol && num_iter_D_T < const.max_iter
        D = D_next;
        T = ne_func.get_T(policy, policy, D, const, param);
        D_next = ne_func.get_D(T, const);
        num_iter_D_T = num_iter_D_T + 1;
    end
    if num_iter_D_T == const.max_iter
        fprintf('Iteration %d did not find convergent (D,T) pair\n', num_iter);
    end
    
    % Step 3-2: Best response of agent i to all other agents j playing policy
    % This is a dynamic progarmming problem solved using policy iteration
    % Step 3-2.0: Initialize agent i's best response policy with current NE guess
    policy_i = policy;
    % Step 3-2.1: Get current stage cost for agent i playing policy_i
    c_i = ne_func.get_c(policy_i, policy, D, const, param);
    % Step 3-2.2: Get probability transition matrix for agent i playing policy_i
    T_i = ne_func.get_T(policy_i, policy, D, const, param);
    % Step 3-2.3: Get expected utility for agent i playing policy_i
    theta_i = (eye(const.num_k) - const.alpha * T_i) \ c_i;
    % Step 3-2.4: Get the expected cost matrix for agent i as per the messages
    % they would transmit, given current expected utility theta_i
    rho_i = ne_func.get_rho(policy, D, theta_i, const, param);
    % Step 3-2.5: Get next minimizing policy for agent i
    policy_i_next = ne_func.get_policy(rho_i, const);
    % Step 3-2.6: Policy iteration - repeat Steps 2.1-2.5 until policy_i
    % converges
    policy_i_error = ne_func.policy_norm(policy_i, policy_i_next, inf);
    num_policy_iter = 0;
    % Display status
    fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_iter, num_policy_iter, policy_i_error);
    while policy_i_error > const.tol && num_policy_iter < const.max_iter
        policy_i = policy_i_next;
        c_i = ne_func.get_c(policy_i, policy, D, const, param);
        T_i = ne_func.get_T(policy_i, policy, D, const, param);
        theta_i = (eye(const.num_k) - const.alpha * T_i) \ c_i;
        rho_i = ne_func.get_rho(policy, D, theta_i, const, param);
        policy_i_next = ne_func.get_policy(rho_i, const);
        policy_i_error = ne_func.policy_norm(policy_i, policy_i_next, inf);
        num_policy_iter = num_policy_iter + 1;

        fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_iter, num_policy_iter, policy_i_error);
    end
    policy_next = policy_i;
    
    policy_error = ne_func.policy_norm(policy, policy_next, inf);
    
    % Display status
    fprintf('Iteration %d policy error %f\n', num_iter, policy_error);
    policy_hist_end = zeros(1, const.num_k);
    policy_hist_end(1) = const.k(policy_next{1}==1);
    fprintf('Iteration %d policy: %d', num_iter, policy_hist_end(1));
    for i_k_i = 2 : const.num_k
        policy_hist_end(i_k_i) = const.k(policy_next{i_k_i}==1);
        fprintf("->%d", policy_hist_end(i_k_i));
    end
    % Detect a limit cycle
    limit_cycle = false;
    for policy_hist_i = 1 : size(policy_hist, 1)
        if isequal(policy_hist(policy_hist_i,:), policy_hist_end)
            % Limit cycle found
            limit_cycle = true;
            policy_limit_cycle = policy_hist(policy_hist_i:end,:);
            policy_limit_cycle_code = policy_limit_cycle * (1 : const.num_k).';
            break;
        end
    end
    policy_hist = [policy_hist; policy_hist_end];
    fprintf('\n\n');
    if limit_cycle
        break;
    end
end

%% Inform user when done
fprintf('DONE\n\n');