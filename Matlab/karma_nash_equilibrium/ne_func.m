classdef ne_func
    % Miscellaneous helper functions used in scripts
    methods(Static)
        % Gets karma transition matrix
        function T = get_T(policy_i, policy_j, D, const, param)
            % T is the transition probability matrix with num_k x num_k entries.
            % Entry T(i_k_i,i_next_k_i) denotes probability of agent i's karma
            % level transitioning from k(i_k_i) to k(i_next_k_i).
            T = zeros(const.num_k);
            for i_k_i = 1 : const.num_k
                for i_next_k_i = 1 : const.num_k
                    % Expectation over u_i
                    for u_i = [0, param.U]
                        p_u_i = 0.5;
                        % Expectation over m_i - comes from policy_i
                        for i_m_i = 1 : i_k_i
                            if u_i == 0
                                p_m_i = (const.k(i_m_i) == 0);
                            else
                                p_m_i = policy_i{i_k_i}(i_m_i);
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
                                    % Expectation over m_j - comes from policy_j
                                    for i_m_j = 1 : i_k_j
                                        if u_j == 0
                                            p_m_j = (const.k(i_m_j) == 0);
                                        else
                                            p_m_j = policy_j{i_k_j}(i_m_j);
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
            left_eig_T_1 = null(eye(const.num_k) - T.');
            % Make sure to return a valid probability distribution (sums to 1)
            if ~isempty(left_eig_T_1)
                D = left_eig_T_1 / sum(left_eig_T_1);
            else
                D = 1 / const.num_k * ones(const.num_k, 1);
            end
        end

        % Gets current stage cost
        function c = get_c(policy_i, policy_j, D, const, param)
            c = zeros(const.num_k, 1);
            for i_k_i = 1 : const.num_k
                % Expectation over u_i
                % Can skip u_i = 0 since cost will be zero
                for u_i = param.U
                    p_u_i = 0.5;
                    % Expectation over m_i - comes from policy_i
                    for i_m_i = 1 : i_k_i
                        if u_i == 0
                            p_m_i = (const.k(i_m_i) == 0);
                        else
                            p_m_i = policy_i{i_k_i}(i_m_i);
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
                                % Expectation over m_j - comes from policy_j
                                for i_m_j = 1 : i_k_j
                                    if u_j == 0
                                        p_m_j = (const.k(i_m_j) == 0);
                                    else
                                        p_m_j = policy_j{i_k_j}(i_m_j);
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
                [~, min_i] = ne_func.multi_mins(rho{i_k_i});
                policy{i_k_i}(min_i) = 1 / length(min_i);
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
    end
end