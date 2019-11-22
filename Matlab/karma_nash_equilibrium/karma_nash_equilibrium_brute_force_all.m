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
policy = cell(const.num_k, 1);

%% Iterative over all pure policies to find Nash Equilibrium
num_iter = 0;
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
                                                fprintf('Iteration %d policy %d->%d->%d->%d->%d->%d->%d->%d->%d->%d->%d->%d->%d\n',...
                                                    num_iter, const.k(1), const.k(i2), const.k(i3), const.k(i4), const.k(i5), const.k(i6), const.k(i7), const.k(i8), const.k(i9), const.k(i10), const.k(i11), const.k(i12), const.k(i13));
                                                
                                                % Stationary distribution, parametrized as a vector with num_k cols.
                                                % Note that it sums to 1
                                                % Initialize uniformly at random
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
                                                
                                                % Step 8: If policy did not
                                                % change, we have found a
                                                % Nash Equilibrium!
                                                policy_error = ne_func.policy_norm(policy, policy_next, inf);
                                                
                                                % Save workspace
                                                save('workspaces/karma_nash_equilibrium_brute_force_all.mat');
                                                
                                                % Termination condition
                                                if policy_error <= const.tol
                                                    break;
                                                end
                                                
                                                num_iter = num_iter + 1;
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

%% Inform user when done
fprintf('DONE\n\n');