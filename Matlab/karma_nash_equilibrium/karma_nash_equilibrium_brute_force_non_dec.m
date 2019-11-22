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
const.tol = 1e-3;

% Maximum number of iterations
const.max_iter = 1000;

%% Karma Nash equilibrium policy calculation
% Policy function, parametrized as a (num_k x num_k) matrix. Entry (i,j)
% denotes probability of transmitting message k(j) when karma level is
% k(i). Note that rows must sum to 1
% Initialize to identity
policy = cell(const.num_k, 1);
for i_k_i = 1 : const.num_k
    policy{i_k_i} = zeros(1, i_k_i);
    policy{i_k_i}(i_k_i) = 1;
end
% load('policy_brute_force_non_dec_0.8.mat');

% Indicate to user where we are
num_iter = 0;
fprintf('Iteration %d policy: %d', num_iter, const.k(policy{1}==1));
for i_k_i = 2 : const.num_k
    fprintf('->%d', const.k(policy{i_k_i}==1));
end
fprintf('\n');

% Stationary distribution, parametrized as a vector with num_k cols.
% Note that it sums to 1
% Initialize uniformly at random
D = rand(const.num_k, 1);
D = D / sum(D);

%% First iteration
% Step 1: Compute T from policy and D
T = ne_func.get_T(policy, policy, D, const, param);

% Step 2: Update D from T
D_next = ne_func.get_D(T, const);

% Step 3: Repeat steps 3-4 until (D,T) pair converge
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

% Step 4: Compute expected current stage cost from policy and D
c = ne_func.get_c(policy, policy, D, const, param);

% Step 5: Compute theta from T and c
theta = (eye(const.num_k) - const.alpha * T) \ c;

% Step 6: Compute rho, the expected utility for each of agent i's
% karma-bid pair, from policy, D and theta
rho = ne_func.get_rho(policy, D, theta, const, param);

% Step 7: Update policy, which is minimzer of rho
policy_next = ne_func.get_policy(rho, const);

% Step 8: Calculate error in policies
policy_error = ne_func.policy_norm(policy, policy_next, inf);

%% Iterate over all 'non-decreasing' pure policies to find Nash Equilibrium
while policy_error > const.tol
    num_iter = num_iter + 1;
    
    % 'Step down' the policy from the end and 'step up' to the top after
    % the 'step down' point
    for i_k_i = const.num_k : -1 : 2
        i_m_i = find(policy{i_k_i} == 1);
        i_m_i_prev = find(policy{i_k_i-1} == 1);
        if i_m_i > i_m_i_prev
            policy{i_k_i}(i_m_i) = 0;
            policy{i_k_i}(i_m_i-1) = 1;
            for i_k_i_2 = i_k_i + 1 : const.num_k
                policy{i_k_i_2} = zeros(1, i_k_i_2);
                policy{i_k_i_2}(i_k_i_2) = 1;
            end
            break;
        end
    end
    
    % Indicate to user where we are
    fprintf('Iteration %d policy: %d', num_iter, const.k(policy{1}==1));
    for i_k_i = 2 : const.num_k
        fprintf('->%d', const.k(policy{i_k_i}==1));
    end
    fprintf('\n');
    
    % Re-initialize D
    D = rand(const.num_k, 1);
    D = D / sum(D);

    % Step 1: Compute T from policy and D
    T = ne_func.get_T(policy, policy, D, const, param);

    % Step 2: Update D from T
    D_next = ne_func.get_D(T, const);

    % Step 3: Repeat steps 3-4 until (D,T) pair converge
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

    % Step 4: Compute expected current stage cost from policy and D
    c = ne_func.get_c(policy, policy, D, const, param);

    % Step 5: Compute theta from T and c
    theta = (eye(const.num_k) - const.alpha * T) \ c;

    % Step 6: Compute rho, the expected utility for each of agent i's
    % karma-bid pair, from policy, D and theta
    rho = ne_func.get_rho(policy, D, theta, const, param);

    % Step 7: Update policy, which is minimzer of rho
    policy_next = ne_func.get_policy(rho, const);
    
    % Step 8: Calculate error in policies
    policy_error = ne_func.policy_norm(policy, policy_next, inf);
    
    % Save workspace
    save('workspaces/karma_nash_equilibrium_brute_force_non_dec.mat');
    
    % Termination condition - cannot step down diagonal anymore if max
    % karma level is bidding no karma
    if find(policy{const.num_k} == 1) == 1
        break;
    end
end

%% Inform user when done
fprintf('DONE\n\n');