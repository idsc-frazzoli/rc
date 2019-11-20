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
const.max_iter = 10;

%% Karma Nash equilibrium policy calculation
% Policy function, parametrized as a (num_k x num_k) matrix. Entry (i,j)
% denotes probability of transmitting message k(j) when karma level is
% k(i). Note that rows must sum to 1
policy = cell(const.num_k, 1);

%% Iterative over all pure policies to find Nash Equilibrium
policy{1} = 1;
for i2 = 1 : 2
    policy{2} = zeros(1, 2);
    policy{2}(i2) = 1;
    for i3 = 1 : 3
        policy{3} = zeros(1, 3);
        policy{3}(i3) = 1;
        for i4 = 1 : 4
            policy{4} = zeros(1, 4);
            policy{4}(i4) = 1;
            for i5 = 1 : 5
                policy{5} = zeros(1, 5);
                policy{5}(i5) = 1;
                for i6 = 1 : 6
                    policy{6} = zeros(1, 6);
                    policy{6}(i6) = 1;
                    for i7 = 1 : 7
                        policy{7} = zeros(1, 7);
                        policy{7}(i7) = 1;
                        for i8 = 1 : 8
                            policy{8} = zeros(1, 8);
                            policy{8}(i8) = 1;
                            for i9 = 1 : 9
                                policy{9} = zeros(1, 9);
                                policy{9}(i9) = 1;
                                for i10 = 1 : 10
                                    policy{10} = zeros(1, 10);
                                    policy{10}(i10) = 1;
                                    for i11 = 1 : 11
                                        policy{11} = zeros(1, 11);
                                        policy{11}(i11) = 1;
                                        for i12 = 1 : 12
                                            policy{12} = zeros(1, 12);
                                            policy{12}(i12) = 1;
                                            for i13 = 1 : 13
                                                policy{13} = zeros(1, 13);
                                                policy{13}(i13) = 1;
                                                   
                                                % Where are we
                                                fprintf("1->%d->%d->%d->%d->%d->%d->%d->%d->%d->%d->%d->%d\n", i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13);
                                                
                                                % Stationary distribution, parametrized as a vector with num_k cols.
                                                % Note that it sums to 1
                                                % Initialize uniformly at random
                                                D = rand(const.num_k, 1);
                                                D = D / sum(D);
                                                
                                                % Step 1: Compute T from policy and D
                                                T = get_T(policy, D, const, param);
                                                
                                                % Step 2: Update D from T
                                                D_next = get_D(T, const);
                                                
                                                % Step 3: Repeat steps 3-4 until (D,T) pair converge
                                                num_iter_D_T = 0;
                                                while norm(D - D_next, inf) > const.tol && num_iter_D_T < const.max_iter
                                                    D = D_next;
                                                    T = get_T(policy, D, const, param);
                                                    D_next = get_D(T, const);
                                                    
                                                    num_iter_D_T = num_iter_D_T + 1;
                                                end
                                                
                                                % Step 4: Compute expected current stage cost from policy and D
                                                c = get_c(policy, D, const, param);
                                                
                                                % Step 5: Compute theta from T and c
                                                theta = (eye(const.num_k) - const.alpha * T) \ c;
                                                
                                                % Step 6: Compute rho, the expected utility for each of agent i's
                                                % karma-bid pair, from policy, D and theta
                                                rho = get_rho(policy, D, theta, const, param);
                                                
                                                % Step 7: Update policy, which is minimzer of rho
                                                policy_next = get_policy(rho, const);
                                                
                                                % Step 8: If policy did not
                                                % change, we have found a
                                                % Nash Equilibrium!
                                                policy_error = policy_norm(policy, policy_next, inf);
                                                if policy_error <= const.tol
                                                    break;
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

%% Helper functions
% Gets karma transition matrix
function T = get_T(policy, D, const, param)
    % T is the transition probability matrix with num_k x num_k entries.
    % Entry T(i_k_i,i_next_k_i) denotes probability of agent i's karma
    % level transitioning from k(i_k_i) to k(i_next_k_i).
    T = zeros(const.num_k);
    for i_k_i = 1 : const.num_k
        for i_next_k_i = 1 : const.num_k
            % Expectation over u_i
            for u_i = [0, param.U]
                p_u_i = 0.5;
                % Expectation over m_i
                for i_m_i = 1 : i_k_i
                    if u_i == 0
                        p_m_i = (const.k(i_m_i) == 0);
                    else
                        p_m_i = policy{i_k_i}(i_m_i);
                    end
                    if p_m_i == 0
                        continue;
                    end
                    % Expectation over u_j
                    for u_j = [0, param.U]
                        p_u_j = 0.5;
                        % Expectation over k_j
                        for i_k_j = 1 : const.num_k
                            p_k_j = D(i_k_j);
                            if p_k_j == 0
                                continue;
                            end
                            % Expectation over m_j
                            for i_m_j = 1 : i_k_j
                                if u_j == 0
                                    p_m_j = (const.k(i_m_j) == 0);
                                else
                                    p_m_j = policy{i_k_j}(i_m_j);
                                end
                                if p_m_j == 0
                                    continue;
                                end
                                
                                % This is where the magic happens
                                % Note that in some cases multiple equally
                                % probable next karma levels are attainable
                                % (when bids are the same)
                                k_next = const.k_next{i_k_i,i_k_j}{i_m_i,i_m_j};
                                p = p_u_i * p_m_i * p_u_j * p_k_j * p_m_j;
                                T(i_k_i,i_next_k_i) = T(i_k_i,i_next_k_i)...
                                    + p * 1 / length(k_next) * sum((k_next == const.k(i_next_k_i)));
                            end
                        end
                    end
                end
            end
        end
    end
end

% Gets stationary distribution of transition matrix T
% This essentially solves D = T*D. The solution is the right eigenvector
% corresponding to eigenvalue 1, or the kernel of (I - T)
function D = get_D(T, const)
    eig_T_1 = null(eye(const.num_k) - T.');
    % Make sure to return a valid probability distribution (sums to 1)
    if ~isempty(eig_T_1)
        D = eig_T_1 / sum(eig_T_1);
    else
        D = 1 / const.num_k * ones(const.num_k, 1);
    end
end

% Gets current stage cost
function c = get_c(policy, D, const, param)
    c = zeros(const.num_k, 1);
    for i_k_i = 1 : const.num_k
        % Expectation over u_i
        % Can skip u_i = 0 since cost will be zero
        for u_i = param.U
            p_u_i = 0.5;
            % Expectation over m_i
            for i_m_i = 1 : i_k_i
                if u_i == 0
                    p_m_i = (const.k(i_m_i) == 0);
                else
                    p_m_i = policy{i_k_i}(i_m_i);
                end
                if p_m_i == 0
                    continue;
                end
                m_i = const.k(i_m_i);
                % Expectation over u_j
                for u_j = [0, param.U]
                    p_u_j = 0.5;
                    % Expectation over k_j
                    for i_k_j = 1 : const.num_k
                        p_k_j = D(i_k_j);
                        if p_k_j == 0
                            continue;
                        end
                        % Expectation over m_j
                        for i_m_j = 1 : i_k_j
                            if u_j == 0
                                p_m_j = (const.k(i_m_j) == 0);
                            else
                                p_m_j = policy{i_k_j}(i_m_j);
                            end
                            if p_m_j == 0
                                continue;
                            end
                            m_j = const.k(i_m_j);

                            % This is where the magic happens
                            if m_i < m_j
                                c_now = u_i;
                            elseif m_i > m_j
                                c_now = 0;
                            else
                                c_now = 0.5 * u_i;
                            end
                            
                            p = p_u_i * p_m_i * p_u_j * p_k_j * p_m_j;
                            c(i_k_i) = c(i_k_i) + p * c_now;
                        end
                    end
                end
            end
        end
    end
end

% Gets rho matrix
function rho = get_rho(policy, D, theta, const, param)
    rho = cell(const.num_k, 1);
    for i_k_i = 1 : const.num_k
        rho{i_k_i} = zeros(1, i_k_i);
        for i_m_i = 1 : i_k_i
            m_i = const.k(i_m_i);
            % Expectation over u_j
            for u_j = [0, param.U]
                p_u_j = 0.5;
                % Expectation over k_j
                for i_k_j = 1 : const.num_k
                    p_k_j = D(i_k_j);
                    % Expectation over m_j
                    for i_m_j = 1 : i_k_j
                        if u_j == 0
                            p_m_j = (const.k(i_m_j) == 0);
                        else
                            p_m_j = policy{i_k_j}(i_m_j);
                        end
                        if p_m_j == 0
                            continue;
                        end
                        m_j = const.k(i_m_j);
                            
                        % This is where the magic happens
                        % Current stage cost
                        if m_i < m_j
                            c_now = param.U;
                        elseif m_i > m_j
                            c_now = 0;
                        else
                            c_now = 0.5 * param.U;
                        end

                        % Next karma with current conditions
                        % Note that in some cases multiple equally
                        % probable next karma levels are attainable
                        % (when bids are the same)
                        k_next = const.k_next{i_k_i,i_k_j}{i_m_i,i_m_j};
                        c_future = theta(const.k == k_next(1)) / length(k_next);
                        for i = 2 : length(k_next)
                            c_future = c_future + theta(const.k == k_next(i)) / length(k_next);
                        end
                        
                        p = p_u_j * p_k_j * p_m_j;
                        rho{i_k_i}(i_m_i) = rho{i_k_i}(i_m_i)...
                            + p * (c_now + const.alpha * c_future);
                    end
                end
            end
        end
    end
end

% Gets the minimizing policy of rho
function policy = get_policy(rho, const)
    policy = cell(const.num_k, 1);
    for i_k_i = 1 : const.num_k
        policy{i_k_i} = zeros(1, i_k_i);
        [~, min_i] = multi_mins(rho{i_k_i});
        policy{i_k_i}(min_i) = 1 / length(min_i);
    end
end

% Returns all minimizers (if there are multiple)
function [min_v, min_i] = multi_mins(input)
    [min_v, min_i] = min(input);
    input(min_i) = realmax;
    [next_min_v, next_min_i] = min(input);
    while next_min_v == min_v
        min_i = [min_i, next_min_i];
        input(next_min_i) = realmax;
        [next_min_v, next_min_i] = min(input);
    end
end

% Computes norm on difference in policy matrices
function pol_norm = policy_norm(policy_1, policy_2, p)
    % Concatenate all differences in 1 vector
    diff_vec = [];
    for i = 1 : size(policy_1, 1)
        diff_vec = [diff_vec, policy_1{i} - policy_2{i}];
    end
    pol_norm = norm(diff_vec, p);
end