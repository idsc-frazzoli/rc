classdef ne_func
    % Miscellaneous helper functions used in scripts
    methods(Static)
        % Gets karma transition matrix
        function T = get_T(policy_i, policy_j, D, ne_param)
            % T is the transition probability matrix with num_k x num_k entries.
            % Entry T(i_k_i,i_next_k_i) denotes probability of agent i's karma
            % level transitioning from k(i_k_i) to k(i_next_k_i).
            T = zeros(ne_param.num_k);
            for i_k_i = 1 : ne_param.num_k
                for i_next_k_i = 1 : ne_param.num_k
                    % Expectation over u_i
                    for i_u_i = 1 : ne_param.num_U
                        u_i = ne_param.U(i_u_i);
                        p_u_i = ne_param.p_U(i_u_i);
                        % Expectation over m_i - comes from policy_i
                        for i_m_i = 1 : i_k_i
                            if u_i == 0
                                p_m_i = (ne_param.k(i_m_i) == 0);
                            else
                                p_m_i = policy_i{i_k_i}(i_m_i);
                            end
                            if p_m_i == 0
                                continue;
                            end
                            % Expectation over u_j
                            for i_u_j = 1 : ne_param.num_U
                                u_j = ne_param.U(i_u_j);
                                p_u_j = ne_param.p_U(i_u_j);
                                % Expectation over k_j
                                for i_k_j = 1 : ne_param.num_k
                                    p_k_j = D(i_k_j);
                                    if p_k_j == 0
                                        continue;
                                    end
                                    % Expectation over m_j - comes from policy_j
                                    for i_m_j = 1 : i_k_j
                                        if u_j == 0
                                            p_m_j = (ne_param.k(i_m_j) == 0);
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
                                        k_next = ne_param.k_next{i_k_i,i_k_j}{i_m_i,i_m_j};
                                        p = p_u_i * p_m_i * p_u_j * p_k_j * p_m_j;
                                        T(i_k_i,i_next_k_i) = T(i_k_i,i_next_k_i)...
                                            + p * 1 / length(k_next) * sum((k_next == ne_param.k(i_next_k_i)));
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        
        % Gets karma transition matrix for state implementation
        function T = get_T_states(policy_i, policy_j, D, ne_param)
            % T is the transition probability matrix with
            % num_states x num_states entries.
            % Entry T(i_x_i,i_next_x_i) denotes probability of agent i's
            % state transitioning from x(i_x_i) to x(i_next_x_i)
            % State [u_i = U(i_u_i), k_i = k(i_k_i)] corresponds to
            % x((i_u_i-1)*num_k+i_k_i), i.e. the first num_k rows/cols are
            % 'non-urgent' states and the next num_k rows/cols are 'urgent'
            % states (for binary urgency)
            T = zeros(ne_param.num_states);
            for i_u_i = 1 : ne_param.num_U
                base_i = (i_u_i - 1) * ne_param.num_k;
                for i_k_i = 1 : ne_param.num_k
                    for i_next_u_i = 1 : ne_param.num_U
                        p_next_u_i = ne_param.p_U(i_next_u_i);
                        base_next_i = (i_next_u_i - 1) * ne_param.num_k;
                        for i_next_k_i = 1 : ne_param.num_k
                            % Expectation over m_i - comes from policy_i
                            for i_m_i = 1 : i_k_i
                                p_m_i = policy_i{base_i+i_k_i}(i_m_i);
                                if p_m_i == 0
                                    continue;
                                end
                                % Expectation over [u_j, k_j]
                                for i_u_j = 1 : ne_param.num_U
                                    base_j = (i_u_j - 1) * ne_param.num_k;
                                    for i_k_j = 1 : ne_param.num_k
                                        p_u_k_j = D(base_j+i_k_j);
                                        if p_u_k_j == 0
                                            continue;
                                        end
                                        % Expectation over m_j - comes from policy_j
                                        for i_m_j = 1 : i_k_j
                                            p_m_j = policy_j{base_j+i_k_j}(i_m_j);
                                            if p_m_j == 0
                                                continue;
                                            end

                                            % This is where the magic happens
                                            % Note that in some cases multiple equally
                                            % probable next karma levels are attainable
                                            % (when bids are the same)
                                            k_next = ne_param.k_next{i_k_i,i_k_j}{i_m_i,i_m_j};
                                            p = p_next_u_i * p_m_i * p_u_k_j * p_m_j;
                                            T(base_i+i_k_i,base_next_i+i_next_k_i) =...
                                                T(base_i+i_k_i,base_next_i+i_next_k_i)...
                                                + p * 1 / length(k_next) * sum((k_next == ne_param.k(i_next_k_i)));
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        
        % Gets stationary distribution of transition matrix T
        % This solves D = T * D. The solution is the right eigenvector
        % corresponding to eigenvalue 1, or the kernel of (I - T)
        function D = get_D(T)
            n = length(T);
            left_eig_T_1 = null(eye(n) - T.');
            % Make sure to return a valid probability distribution (sums to 1)
            if ~isempty(left_eig_T_1)
                D = left_eig_T_1 / sum(left_eig_T_1);
            else
                D = 1 / n * ones(n, 1);
            end
        end
        
        % Gets current stage cost
        function c = get_c(policy_i, policy_j, D, ne_param)
            c = zeros(ne_param.num_k, 1);
            for i_k_i = 1 : ne_param.num_k
                % Expectation over u_i
                % Can skip u_i = 0 since cost will be zero
                for u_i = ne_param.u_high
                    p_u_i = ne_param.p_U(end);
                    %p_u_i = 1;
                    % Expectation over m_i - comes from policy_i
                    for i_m_i = 1 : i_k_i
                        if u_i == 0
                            p_m_i = (ne_param.k(i_m_i) == 0);
                        else
                            p_m_i = policy_i{i_k_i}(i_m_i);
                        end
                        if p_m_i == 0
                            continue;
                        end
                        m_i = ne_param.k(i_m_i);
                        % Expectation over u_j
                        for i_u_j = 1 : ne_param.num_U
                            u_j = ne_param.U(i_u_j);
                            p_u_j = ne_param.p_U(i_u_j);
                            % Expectation over k_j
                            for i_k_j = 1 : ne_param.num_k
                                p_k_j = D(i_k_j);
                                if p_k_j == 0
                                    continue;
                                end
                                % Expectation over m_j - comes from policy_j
                                for i_m_j = 1 : i_k_j
                                    if u_j == 0
                                        p_m_j = (ne_param.k(i_m_j) == 0);
                                    else
                                        p_m_j = policy_j{i_k_j}(i_m_j);
                                    end
                                    if p_m_j == 0
                                        continue;
                                    end
                                    m_j = ne_param.k(i_m_j);

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

        % Gets current stage cost for state implementation
        function c = get_c_states(policy_i, policy_j, D, ne_param)
            c = zeros(ne_param.num_states, 1);
            for i_u_i = 1 : ne_param.num_U
                u_i = ne_param.U(i_u_i);
                base_i = (i_u_i - 1) * ne_param.num_k;
                for i_k_i = 1 : ne_param.num_k
                    % Expectation over m_i - comes from policy_i
                    for i_m_i = 1 : i_k_i
                        p_m_i = policy_i{base_i+i_k_i}(i_m_i);
                        if p_m_i == 0
                            continue;
                        end
                        m_i = ne_param.k(i_m_i);
                        % Expectation over [u_j, k_j]
                        for i_u_j = 1 : ne_param.num_U
                            base_j = (i_u_j - 1) * ne_param.num_k;
                            for i_k_j = 1 : ne_param.num_k
                                p_u_k_j = D(base_j+i_k_j);
                                if p_u_k_j == 0
                                    continue;
                                end
                                % Expectation over m_j - comes from policy_j
                                for i_m_j = 1 : i_k_j
                                    p_m_j = policy_j{base_j+i_k_j}(i_m_j);
                                    if p_m_j == 0
                                        continue;
                                    end
                                    m_j = ne_param.k(i_m_j);

                                    % This is where the magic happens
                                    if m_i < m_j
                                        c_now = u_i;
                                    elseif m_i > m_j
                                        c_now = 0;
                                    else
                                        c_now = 0.5 * u_i;
                                    end

                                    p = p_m_i * p_u_k_j * p_m_j;
                                    c(base_i+i_k_i) = c(base_i+i_k_i)...
                                        + p * c_now;
                                end
                            end
                        end
                    end
                end
            end
        end
        
        % Gets rho matrix
        function rho = get_rho(policy, D, theta, ne_param)
            rho = cell(ne_param.num_k, 1);
            for i_k_i = 1 : ne_param.num_k
                rho{i_k_i} = zeros(1, i_k_i);
                for i_m_i = 1 : i_k_i
                    m_i = ne_param.k(i_m_i);
                    % Expectation over u_j
                    for i_u_j = 1 : ne_param.num_U
                        u_j = ne_param.U(i_u_j);
                        p_u_j = ne_param.p_U(i_u_j);
                        % Expectation over k_j
                        for i_k_j = 1 : ne_param.num_k
                            p_k_j = D(i_k_j);
                            % Expectation over m_j
                            for i_m_j = 1 : i_k_j
                                if u_j == 0
                                    p_m_j = (ne_param.k(i_m_j) == 0);
                                else
                                    p_m_j = policy{i_k_j}(i_m_j);
                                end
                                if p_m_j == 0
                                    continue;
                                end
                                m_j = ne_param.k(i_m_j);

                                % This is where the magic happens
                                % Current stage cost
                                if m_i < m_j
                                    c_now = ne_param.u_high;
                                elseif m_i > m_j
                                    c_now = 0;
                                else
                                    c_now = 0.5 * ne_param.u_high;
                                end

                                % Next karma with current conditions
                                % Note that in some cases multiple equally
                                % probable next karma levels are attainable
                                % (when bids are the same)
                                k_next = ne_param.k_next{i_k_i,i_k_j}{i_m_i,i_m_j};
                                c_future = theta(ne_param.k == k_next(1)) / length(k_next);
                                for i = 2 : length(k_next)
                                    c_future = c_future + theta(ne_param.k == k_next(i)) / length(k_next);
                                end

                                p = p_u_j * p_k_j * p_m_j;
                                rho{i_k_i}(i_m_i) = rho{i_k_i}(i_m_i)...
                                    + p * (c_now + ne_param.alpha * c_future);
                            end
                        end
                    end
                end
            end
        end
        
        % Gets rho matrix for state implementation
        function rho = get_rho_states(policy, D, theta, ne_param)
            rho = cell(ne_param.num_states, 1);
            for i_u_i = 1 : ne_param.num_U
                u_i = ne_param.U(i_u_i);
                base_i = (i_u_i - 1) * ne_param.num_k;
                for i_k_i = 1 : ne_param.num_k
                    rho{base_i+i_k_i} = zeros(1, i_k_i);
                    for i_m_i = 1 : i_k_i
                        m_i = ne_param.k(i_m_i);
                        % Expectation over [u_j, k_j]
                        for i_u_j = 1 : ne_param.num_U
                            base_j = (i_u_j - 1) * ne_param.num_k;
                            for i_k_j = 1 : ne_param.num_k
                                p_u_k_j = D(base_j+i_k_j);
                                % Expectation over m_j
                                for i_m_j = 1 : i_k_j
                                    p_m_j = policy{base_j+i_k_j}(i_m_j);
                                    if p_m_j == 0
                                        continue;
                                    end
                                    m_j = ne_param.k(i_m_j);

                                    % This is where the magic happens
                                    % Current stage cost
                                    if m_i < m_j
                                        c_now = u_i;
                                    elseif m_i > m_j
                                        c_now = 0;
                                    else
                                        c_now = 0.5 * u_i;
                                    end

                                    % Next karma with current conditions
                                    % Note that in some cases multiple equally
                                    % probable next karma levels are attainable
                                    % (when bids are the same)
                                    next_k_i = ne_param.k_next{i_k_i,i_k_j}{i_m_i,i_m_j};
                                    num_next_k_i = length(next_k_i);
                                    c_future = 0;
                                    for i_next_u_i = 1 : ne_param.num_U
                                        p_next_u_i = ne_param.p_U(i_next_u_i);
                                        base_next_i = (i_next_u_i - 1) * ne_param.num_k;
                                        for i = 1 : num_next_k_i
                                            p_next_k_i = 1 / num_next_k_i;
                                            i_next_k_i = find(ne_param.k == next_k_i(i));
                                            p_next_i = p_next_u_i * p_next_k_i;
                                            c_future = c_future + p_next_i * theta(base_next_i+i_next_k_i);
                                        end
                                    end

                                    p = p_u_k_j * p_m_j;
                                    rho{base_i+i_k_i}(i_m_i) =...
                                        rho{base_i+i_k_i}(i_m_i)...
                                        + p * (c_now + ne_param.alpha * c_future);
                                end
                            end
                        end
                    end
                end
            end
        end
        
        % Gets the minimizing policy of rho
        function policy = get_policy(rho)
            n = size(rho, 1);
            policy = cell(n, 1);
            for i = 1 : n
                policy{i} = zeros(1, length(rho{i}));
                [~, min_i] = ne_func.multi_mins(rho{i});
                policy{i}(min_i) = 1 / length(min_i);
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
        
        % Plot policy
        function plot_policy(fg, position, policy, ne_param, title, colormap)
            policy_mat = nan(ne_param.num_k);
            for i_k_i = 1 : ne_param.num_k
                policy_mat(i_k_i,1:i_k_i) = policy{i_k_i};
            end
            policy_mat(policy_mat == 0) = nan;
            figure(fg);
            fig = gcf;
            fig.Position = position;
            h = heatmap(ne_param.k, ne_param.k, policy_mat.', 'ColorbarVisible','off');
            h.YDisplayData = flipud(h.YDisplayData);
            h.Title = title;
            h.XLabel = 'Karma';
            h.YLabel = 'Message';
            h.FontName = 'Ubuntu';
            h.FontSize = 12;
            if exist('colormap', 'var')
                h.Colormap = colormap;
            end
            h.ColorLimits = [0 1];
            h.CellLabelFormat = '%.2f';
            drawnow;
        end
        
        % Plot policy for state implementation
        function plot_policy_states(fg, position, policy, ne_param, title, colormap)
            figure(fg);
            fig = gcf;
            fig.Position = position;
            for i_u_i = 1 : ne_param.num_U
                base_i = (i_u_i - 1) * ne_param.num_k;
                policy_mat = nan(ne_param.num_k);
                for i_k_i = 1 : ne_param.num_k
                    policy_mat(i_k_i,1:i_k_i) = policy{base_i+i_k_i};
                end
                policy_mat(policy_mat == 0) = nan;
                subplot(1,ne_param.num_U,i_u_i);
                h = heatmap(ne_param.k, ne_param.k, policy_mat.', 'ColorbarVisible','off');
                h.YDisplayData = flipud(h.YDisplayData);
                h.Title = [title, ' for u = ', num2str(ne_param.U(i_u_i))];
                h.XLabel = 'Karma';
                h.YLabel = 'Message';
                h.FontName = 'Ubuntu';
                h.FontSize = 10;
                if exist('colormap', 'var')
                    h.Colormap = colormap;
                end
                h.ColorLimits = [0 1];
                h.CellLabelFormat = '%.2f';
            end
            drawnow;
        end
        
        % Plot stationary distribution D
        function plot_D(fg, position, D, ne_param, title)
            figure(fg);
            fig = gcf;
            fig.Position = position;
            bar(ne_param.k, D);
            axes = gca;
            axis tight;
            axes.Title.FontName = 'ubuntu';
            axes.Title.String = title;
            axes.Title.FontSize = 12;
            axes.XAxis.FontSize = 10;
            axes.YAxis.FontSize = 10;
            axes.XLabel.FontName = 'ubuntu';
            axes.XLabel.String = 'Karma';
            axes.XLabel.FontSize = 12;
            axes.YLabel.FontName = 'ubuntu';
            axes.YLabel.String = 'Probability';
            axes.YLabel.FontSize = 12;
            drawnow;
        end
        
        % Plot stationary distribution D for state implementation
        function plot_D_states(fg, position, D, ne_param, title)
            figure(fg);
            fig = gcf;
            fig.Position = position;
            for i_u_i = 1 : ne_param.num_U
                base_i = (i_u_i - 1) * ne_param.num_k;
                subplot(1,ne_param.num_U,i_u_i);
                bar(ne_param.k, D(base_i+1:base_i+ne_param.num_k));
                axes = gca;
                axis tight;
                axes.Title.FontName = 'ubuntu';
                axes.Title.String = [title, ' for u = ', num2str(ne_param.U(i_u_i))];
                axes.Title.FontSize = 12;
                axes.XAxis.FontSize = 10;
                axes.YAxis.FontSize = 10;
                axes.XLabel.FontName = 'ubuntu';
                axes.XLabel.String = 'Karma';
                axes.XLabel.FontSize = 12;
                axes.YLabel.FontName = 'ubuntu';
                axes.YLabel.String = 'Probability';
                axes.YLabel.FontSize = 12;
            end
            drawnow;
        end
    end
end