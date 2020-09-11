classdef func
    % Functions used in scripts
    methods(Static)
        % Allocates cost matrix
        % Values are initialized with -1 to tell when agents did not
        % participate in an interaction
        function c = allocate_cost(param)
            c = nan(param.max_num_inter_per_agent, param.N);
        end
        
        % Gets stationary distribution of transition matrix T
        % This essentially solves d = T.'*d. The solution is the left eigenvector
        % corresponding to eigenvalue 1, or the kernel of (I - T.')
        function d = stat_dist(T)
            n = size(T, 1);
            left_eig_T_1 = null(eye(n) - T.');
            % Make sure to return a valid probability distribution (sums to 1)
            if ~isempty(left_eig_T_1)
                d = left_eig_T_1 / sum(left_eig_T_1);
            else
                d = 1 / n * ones(n, 1);
            end
        end
        
        % Returns all maximizers (if there are multiple)
        function [v_max, i_max] = multi_maxes(input)
            [v_max, i_max] = max(input);
            input(i_max) = -realmax;
            [next_v_max, next_i_max] = max(input);
            while next_v_max == v_max
                i_max = [i_max, next_i_max];
                input(next_i_max) = -realmax;
                [next_v_max, next_i_max] = max(input);
            end
        end

        % Checks if there are multiple maximizers and returns one uniformly at
        % random
        function [v_max, i_max] = multi_max(input)
            % Get maximizers
            [v_max, i_max] = func.multi_maxes(input);
            % Choose one uniformly at random if there are multiple
            num_max = length(i_max);
            if num_max > 1
                i_max = datasample(i_max, 1);
            end
        end
        
        % Gets uniform distribution over karma adjusted to have correct
        % average karma k_ave for infinite population N
        function [s_up_k_uniform, K_uniform] = get_s_up_k_uniform(k_ave)
            K_uniform = (0 : k_ave * 2).';
            num_K = length(K_uniform);
            s_up_k_uniform = 1 / num_K * ones(num_K, 1);
        end
        
        % Gets a karma initialization for finite N agents that is close in
        % distribution to specified infinite N distribution and has the
        % correct k_ave/k_tot
        function init_k = get_init_k(s_up_k_inf, K, param)
            % Get finite N distribution close to infinite N distribution
            % and with correct k_ave/k_tot
            s_up_k_N = round(param.N * s_up_k_inf);
            s_up_k_N = reshape(s_up_k_N, 1, []);
            missing_agents = param.N - sum(s_up_k_N);
            missing_karma = param.k_tot - s_up_k_N * K;
            while missing_agents ~= 0 || missing_karma ~= 0
                if missing_agents ~= 0
                    % Need to adjust agent count (and possibly karma)
                    karma_to_adjust = min([floor(missing_karma / missing_agents), K(end)]);
                    if karma_to_adjust >= 0
                        % Need to either add both agents and karma or
                        % remove both agents and karma
                        i_karma_to_adjust = find(K == karma_to_adjust);
                        agents_to_adjust = missing_agents - rem(missing_karma, missing_agents);
                        if agents_to_adjust < 0 && s_up_k_N(i_karma_to_adjust) == 0
                            % Need to remove agents from a karma that
                            % doesn't have agents. Find closest karma with
                            % agents
                            karma_with_agents = K(s_up_k_N > 0);
                            [~, i_closest_karma_with_agents] = min(abs(karma_with_agents - K(i_karma_to_adjust)));
                            i_karma_to_adjust = find(K == karma_with_agents(i_closest_karma_with_agents));
                        end
                        s_up_k_N(i_karma_to_adjust) = max([s_up_k_N(i_karma_to_adjust) + agents_to_adjust, 0]);
                    else
                        if missing_agents > 0
                            % Need to add agents and remove karma
                            % First remove one agent with the closest
                            % amount of karma to what needs to be removed
                            i_karma_to_remove = find(K == min([abs(missing_karma), K(end)]));
                            if s_up_k_N(i_karma_to_remove) == 0
                                karma_with_agents = K(s_up_k_N > 0);
                                [~, i_closest_karma_with_agents] = min(abs(karma_with_agents - K(i_karma_to_remove)));
                                i_karma_to_remove = find(K == karma_with_agents(i_closest_karma_with_agents));
                            end
                            s_up_k_N(i_karma_to_remove) = s_up_k_N(i_karma_to_remove) - 1;
                            % Now add the required amount of agents with
                            % zero karma to not change karma count
                            s_up_k_N(1) = s_up_k_N(1) + missing_agents + 1;
                        else
                            % Need to remove agents and add karma
                            % First remove agents with least amount of
                            % karma, and keep track of amount of karma
                            % removed in the process. Remove one extra
                            % agent because one will be added with the
                            % required amount of karma
                            agents_to_remove = abs(missing_agents) + 1;
                            agents_removed = 0;
                            karma_removed = 0;
                            i_karma_to_remove = 1;
                            while agents_removed < agents_to_remove && i_karma_to_remove <= length(K)
                                agents_can_remove = min(agents_to_remove - agents_removed, s_up_k_N(i_karma_to_remove));
                                s_up_k_N(i_karma_to_remove) = s_up_k_N(i_karma_to_remove)- agents_can_remove;
                                agents_removed = agents_removed + agents_can_remove;
                                karma_removed = karma_removed + agents_can_remove * K(i_karma_to_remove);
                                i_karma_to_remove = i_karma_to_remove + 1;
                            end
                            % Now add one agent with the required amount of
                            % karma
                            i_karma_to_add = find(K == min([missing_karma + karma_removed, K(end)]));
                            s_up_k_N(i_karma_to_add) = s_up_k_N(i_karma_to_add) + 1;
                        end
                    end
                else
                    if missing_karma > 0
                        % Need to add karma only
                        % Remove one agent with least karma and add one
                        % with the required amount of karma
                        karma_with_agents = K(s_up_k_N > 0);
                        i_karma_to_remove = find(K == karma_with_agents(1));
                        s_up_k_N(i_karma_to_remove) = s_up_k_N(i_karma_to_remove) - 1;
                        i_karma_to_add = find(K == min([missing_karma + K(i_karma_to_remove), K(end)]));
                        s_up_k_N(i_karma_to_add) = s_up_k_N(i_karma_to_add) + 1;
                    else
                        % Need to remove karma only
                        % Remove one agent with the closest amount of karma
                        % to what needs to be removed and add one agent
                        % with zero karma
                        i_karma_to_remove = find(K == min([abs(missing_karma), K(end)]));
                        if s_up_k_N(i_karma_to_remove) == 0
                            karma_with_agents = K(s_up_k_N > 0);
                            [~, i_closest_karma_with_agents] = min(abs(karma_with_agents - K(i_karma_to_remove)));
                            i_karma_to_remove = find(K == karma_with_agents(i_closest_karma_with_agents));
                        end
                        s_up_k_N(i_karma_to_remove) = s_up_k_N(i_karma_to_remove) - 1;
                        s_up_k_N(1) = s_up_k_N(1) + 1;
                    end
                end
                missing_agents = param.N - sum(s_up_k_N);
                missing_karma = param.k_tot - s_up_k_N * K;
            end
            
            % Initialize karma for N agents as per finite N distribution
            init_k = zeros(1, param.N);
            start_i = 0;
            for i_k = 1 : length(K)
                num_agents = s_up_k_N(i_k);
                init_k(start_i+1:start_i+num_agents) = K(i_k);
                start_i = start_i + num_agents;
            end
            
%             % Shuffle initial karma in the end because why not
%             init_k = init_k(randperm(param.N));
        end
        
        % Allocates karma matrix and sets the initial karma as per init_k
        % Values are initialized with -1 to tell when agents did not
        % participate in an interaction
        function k = allocate_karma(param, init_k)
            k = nan(param.max_num_inter_per_agent, param.N);
            k(1,:) = init_k;
        end
        
        % Gets pure policy from mixed policy using a threshold
        function pi_pure = get_pure_policy(pi, K, param)
            num_K = length(K);
            pi_pure = nan(param.num_U, num_K);
            pi_pure(:,1) = 0;
            for i_u = 1 : param.num_U
                for i_k = 2 : num_K
                    i_max = 1 : i_k;
                    [pi_max, i_pi_max] = max(pi(i_u,i_k,i_max));
                    i_max(i_pi_max) = [];
                    pi_max_2 = max(pi(i_u,i_k,i_max));
                    if pi_max_2 / pi_max < param.pure_policy_tol
                        pi_pure(i_u,i_k) = K(i_pi_max);
                    end
                end
            end
        end
        
        % Accumulates the cost in the whole history up to now
        function a = accumulate_cost(c, agents_id, u, num_inter, param)
            a = zeros(1, length(agents_id));
            for i_agent = 1 : length(agents_id)
                id = agents_id(i_agent);
                a(i_agent) = sum(c(1:num_inter(id)-1,id)) + u(i_agent);
                if param.normalize_cost
                    a(i_agent) = a(i_agent) / num_inter(id);
                end
            end
        end
        
        % Find the relative cost to urgency in the history up to past
        % horizon m. m = inf means all the history
        function r = relative_cost(c, agents_id, u_hist, num_inter, m)
            r = zeros(1, length(agents_id));
            for i_agent = 1 : length(agents_id)
                id = agents_id(i_agent);
                if m == inf
                    start_i = 1;
                else
                    start_i = max([num_inter(id) - m, 1]);
                end
                r(i_agent) = sum(c(start_i:num_inter(id)-1,id)) + u_hist(num_inter(id),id);
%                 u_mean = mean(u_hist(1:num_inter(id),id));
% %                 u_mean = 1.5;
%                 r(i_agent) = r(i_agent) / (min([m + 1, num_inter(id)]) * u_mean);
%                 if m == inf
%                     r(i_agent) = r(i_agent) / sum(u_hist(1:num_inter(id),id));
%                 end
                r(i_agent) = r(i_agent) / sum(u_hist(start_i:num_inter(id),id));
            end
        end
        
        % Gets agents that lose given indeces of agents that win
        function lose_id = get_lose_id(agents_id, i_win)
            lose_id = agents_id;
            lose_id(i_win) = [];
        end
        
        % Gets karma paid by winning agent to losing agent
        function p = get_karma_payment(m_win, m_lose, param)
            switch param.m_exchange
                case 0      % Pay as bid
                    p = m_win;
                case 1      % Pay difference
                    p = m_win - m_lose;
                case 2      % Pay difference and pay one on tie
                    if m_win == m_lose && m_win >= 1
                        p = 1;
                    else
                        p = m_win - m_lose;
                    end
            end
        end
        
        % Gets karma at the end of the experiment
        function end_k = get_end_karma(k, num_inter, param)
            end_k = zeros(1, param.N);
            for i_agent = 1 : param.N
                end_k = k(num_inter(i_agent)+1,i_agent);
            end
        end

        % Gets accumulated costs counting in warm-up period reset
        function a = get_accumulated_cost_old(c, param)
            a = [cumsum(c(1:param.t_warm_up,:)); cumsum(c(param.t_warm_up+1:end,:))];
        end
        
        % Gets accumulated costs
        function a = get_accumulated_cost(c, num_inter, param)
            a = nan(max(num_inter), param.N);
            for i_agent = 1 : param.N
                a(1:num_inter(i_agent),i_agent) = cumsum(c(1:num_inter(i_agent),i_agent));
                if param.normalize_cost
                    a(1:num_inter(i_agent),i_agent) = a(1:num_inter(i_agent),i_agent) ./ (1 : num_inter(i_agent)).';
                end
                % Fill the end of the matrix up with the last total
                % accumulated cost
                a(num_inter(i_agent)+1:end,i_agent) = a(num_inter(i_agent),i_agent);
            end
        end
        
        % Gets relative costs
        function r = get_relative_cost(c, u_hist, num_inter, param)
            r = nan(max(num_inter), param.N);
            for i_agent = 1 : param.N
                r(1:num_inter(i_agent),i_agent) = cumsum(c(1:num_inter(i_agent),i_agent))...
                    ./ cumsum(u_hist(1:num_inter(i_agent),i_agent));
                % Fill the end of the matrix up with the last relative cost
                r(num_inter(i_agent)+1:end,i_agent) = r(num_inter(i_agent),i_agent);
            end
        end
        
        % Gets service ratios
        function s = get_service_ratio(c, u_hist, num_inter, param)
            s = nan(max(num_inter), param.N);
            for i_agent = 1 : param.N
                s(1:num_inter(i_agent),i_agent) = cumsum(u_hist(1:num_inter(i_agent),i_agent) - c(1:num_inter(i_agent),i_agent))...
                    ./ cumsum(u_hist(1:num_inter(i_agent),i_agent));
                % Fill the end of the matrix up with the last relative cost
                s(num_inter(i_agent)+1:end,i_agent) = s(num_inter(i_agent),i_agent);
            end
        end
        
        % Gets the empirical karma distribution for the whole society and
        % per agent
        function [k_dist, k_dist_agents] = get_karma_dist(k, param)
            k_min = nanmin(k(:));
            k_max = nanmax(k(:));
            K = k_min : k_max;
            num_K = length(K);
            k_dist = zeros(num_K, 1);
            k_dist_agents = zeros(num_K, param.N);
            for i_k = 1 : num_K
                k_dist(i_k) = length(find(k(:) == K(i_k)));
                for i_agent = 1 : param.N
                    k_dist_agents(i_k,i_agent) = length(find(k(:,i_agent) == K(i_k)));
                end
            end
            k_dist = k_dist / sum(k_dist);
            k_dist_agents = k_dist_agents ./ sum(k_dist_agents, 1);
        end
        
        % Gets entropy of accumulated costs with limited memory
        function ent = get_entropy_lim_mem(c, num_inter, lim_mem_steps, param)
            % Get costs accumulated over the limited memory steps for all
            % timesteps and all agents
            a = [];
            for i_agent = 1 : param.N
                num_inter_agent = num_inter(i_agent);
                num_hist = max([num_inter_agent - lim_mem_steps + 1, 1]);
%                 num_hist = num_inter_agent;
                a_agent = zeros(num_hist, 1);
                for i_hist = num_hist : -1 : 1
                    end_i = i_hist + num_inter_agent - num_hist;
                    start_i = max([end_i - lim_mem_steps + 1, 1]);
                    a_agent(i_hist) = sum(c(start_i:end_i,i_agent));
                end
                a = [a; a_agent];
            end
            
            % Get distribution of the accumulated costs
            a_unique = unique(a);
            num_a_unique = length(a_unique);
            a_dist = zeros(num_a_unique, 1);
            for i_a = 1 : num_a_unique
                a_dist(i_a) = length(find(a == a_unique(i_a)));
            end
            a_dist = a_dist / sum(a_dist);

            % Get entropy from distribution
            ent = func.get_entropy(a_dist);
        end
        
        % Gets the entropy of the given distribution
        function ent = get_entropy(dist)
            dist(dist == 0) = [];
            ent = -sum(dist .* log2(dist));
        end
        
        % Standardizes input distribution using standardization method
        % specified in param
        function a_std = get_standardized_cost(a, a_mean, a_var, param)
            switch param.standardization_method
                % 0-mean 1-variance standardization
                case 0
                    a_std = func.standardize_mean_var(a, a_mean, a_var);
                % Order ranking standardization
                case 1
                    a_std = func.order_rank(a);
                % normalized order ranking standardization, i.e. order ranking scaled
                % between 0-1
                case 2
                    a_std = func.order_rank_norm(a);
            end
        end
        
        % Standardizes input distribution given mean and variance vectors
        function output = standardize_mean_var(input, input_mean, input_var)
            output = (input - input_mean) ./ sqrt(input_var);
            output(isnan(output)) = 0;
        end
        
        % Computes the oder ranking of the input
        function output = order_rank(input)
            output = zeros(size(input));
            for i = 1 : size(input, 1)
                output(i,:) = tiedrank(input(i,:));
            end
            output(isnan(output)) = 0;
        end
        
        % Computes the normalized oder ranking of the input, which is
        % scaled between 0 and 1
        function output = order_rank_norm(input)
            output = zeros(size(input));
            v_min = 1;
            v_max = size(output, 2);
            for i = 1 : size(input, 1)
                output(i,:) = tiedrank(input(i,:));
                output(i,:) = (output(i,:) - v_min) / (v_max - v_min);
            end
            output(isnan(output)) = 0;
        end
        
        % Computes the autocorrelation of signal
        function [acorr, tau] = autocorrelation(input)
            T = size(input, 1);
            N = size(input, 2);
            center_t = ceil(T / 2);
            tau = -center_t + 1 : 1 : center_t;
            mult_mat = input * input.';
            acorr = zeros(1, T);
            % Salvage symmetry of autocorrelation about zero and calculate it for
            % positive time shifts only
            tau_0_i = find(tau == 0);
            for i = tau_0_i : T
                acorr(i) = sum(diag(mult_mat, tau(i)));
            end
            for i = 1 : tau_0_i - 1
                acorr(i) = acorr(end-i);
            end
            acorr = acorr ./ ((T - abs(tau)) * N);
        end
        
        % Plot karma distribution
        function plot_karma_dist(fg, position, k_dist, K, k_ave, alpha)
            figure(fg);
            fig = gcf;
            fig.Position = position;
            k_dist_mean = mean(k_dist);
            k_dist_std = std(k_dist, 1, 1);
            bar(K, k_dist_mean);
            hold on;
%             errorbar(K, k_dist_mean, k_dist_std, '--', 'LineWidth', 2);
            axis tight;
            axes = gca;
            axes.Title.FontName = 'ubuntu';
            axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' karma distribution'];
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
        
        % Wrtie performance comparison to csv file
        function write_performance_comparison_csv(alpha, IE_ne, UF_ne, IE_bid_all, UF_bid_all, IE_sw, UF_sw, IE_u, IE_rand, fileprefix)
            % Populate column vectors appropriately
            alpha = reshape(alpha, [], 1);
            num_alpha = length(alpha);
            e = zeros(num_alpha, 1);
            f = zeros(num_alpha, 1);
            for i_alpha = 1 : num_alpha
                e(i_alpha) = -IE_ne{i_alpha}(end);
                f(i_alpha) = -UF_ne{i_alpha}(end);
            end
            if ~any(alpha == 0)
                alpha = [alpha; 0];
                e = [e; -IE_bid_all(end)];
                f = [f; -UF_bid_all(end)];
                num_alpha = length(alpha);
            end            
            
            % Make vectors out of e_sw, f_sw, e_opt, e_rand
            e_sw = -IE_sw(end) * ones(num_alpha, 1);
            f_sw = -UF_sw(end) * ones(num_alpha, 1);
            e_opt = -IE_u(end) * ones(num_alpha, 1);
            e_rand = -IE_rand(end) * ones(num_alpha, 1);
            
            % PoK
            PoK = e ./ e_opt;
            PoK_sw = e_sw ./ e_opt;
            PoK_opt = e_opt ./ e_opt;
            PoK_rand = e_rand ./ e_opt;
            
            % Header
            header = ["alpha", "e", "PoK", "f", "e_sw", "PoK_sw", "f_sw", "e_opt", "PoK_opt", "e_rand", "PoK_rand"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            data = [alpha, e, PoK, f, e_sw, PoK_sw, f_sw, e_opt, PoK_opt, e_rand, PoK_rand];
            dlmwrite(filename, data, '-append');
        end
        
        % Wrtie performance comparison of centralized policies to csv file
        function write_performance_comparison_cent_pol_csv(IE_rand, UF_rand, IE_u, UF_u, IE_a, UF_a, IE_u_a, UF_u_a, fileprefix)
            % Convention
            % 1 => baseline random
            % 2 => centralized urgency
            % 3 => centralized cost
            % 4 => centralized urgency then cost
            num_cent_pol = 4;
            cent_pol = (1 : num_cent_pol).';
            
            % Populate column vectors appropriately
            e = zeros(num_cent_pol, 1);
            f = zeros(num_cent_pol, 1);
            e(1) = -IE_rand(end);
            e(2) = -IE_u(end);
            e(3) = -IE_a(end);
            e(4) = -IE_u_a(end);
            f(1) = -UF_rand(end);
            f(2) = -UF_u(end);
            f(3) = -UF_a(end);
            f(4) = -UF_u_a(end);
            
            % Make vector out of e_opt
            e_opt = -IE_u(end) * ones(num_cent_pol, 1);
            
            % PoK
            PoK = e ./ e_opt;
            PoK_opt = e_opt ./ e_opt;
            
            % Header
            header = ["cent_pol", "e", "PoK", "f", "e_opt", "PoK_opt"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            data = [cent_pol, e, PoK, f, e_opt, PoK_opt];
            dlmwrite(filename, data, '-append');
        end
        
        % Wrtie performance comparison of limited memory policies to csv file
        function write_performance_comparison_lim_mem_csv(lim_mem_steps, IE_lim_mem_a, UF_lim_mem_a, IE_u, fileprefix)
            % Populate column vectors appropriately
            lim_mem_steps = reshape(lim_mem_steps, [], 1);
            num_lim_mem_steps = length(lim_mem_steps);
            e = zeros(num_lim_mem_steps, 1);
            f = zeros(num_lim_mem_steps, 1);
            for i_lim_mem = 1 : num_lim_mem_steps
                e(i_lim_mem) = -IE_lim_mem_a{i_lim_mem}(end);
                f(i_lim_mem) = -UF_lim_mem_a{i_lim_mem}(end);
            end
            
            % Make vector out of e_opt
            e_opt = -IE_u(end) * ones(num_lim_mem_steps, 1);
            
            % PoK
            PoK = e ./ e_opt;
            PoK_opt = e_opt ./ e_opt;
            
            % Header
            header = ["m", "e", "PoK", "f", "e_opt", "PoK_opt"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            data = [lim_mem_steps, e, PoK, f, e_opt, PoK_opt];
            dlmwrite(filename, data, '-append');
        end
        
        % Wrtie fairness vs. alpha to csv file
        function write_fairness_alpha_csv(alpha, UF_ne, UF_bid_all, UF_sw, UF_u, UF_rand, UF_a, lim_mem_steps, UF_lim_mem_a, fileprefix)
            % Populate column vectors appropriately
            alpha = reshape(alpha, [], 1);
            num_alpha = length(alpha);
            f = zeros(num_alpha, 1);
            for i_alpha = 1 : num_alpha
                f(i_alpha) = -UF_ne{i_alpha}(end);
            end
            if ~any(alpha == 0)
                alpha = [alpha; 0];
                f = [f; -UF_bid_all(end)];
                num_alpha = length(alpha);
            end            
            
            % Make vectors out of fairness of benchmark policies
            f_sw = -UF_sw(end) * ones(num_alpha, 1);
            f_u = -UF_u(end) * ones(num_alpha, 1);
            f_rand = -UF_rand(end) * ones(num_alpha, 1);
            f_a = -UF_a(end) * ones(num_alpha, 1);
            
            % Fairness of limited memory policies
            num_lim_mem_steps = length(lim_mem_steps);
            f_lim_mem = zeros(num_alpha, num_lim_mem_steps);
            for i_lim_mem = 1 : num_lim_mem_steps
                f_lim_mem(:,i_lim_mem) = -UF_lim_mem_a{i_lim_mem}(end) * ones(num_alpha, 1);
            end
            
            % Header
            header = ["alpha", "f", "f_sw", "f_u", "f_rand", "f_a"];
            for i_lim_mem = 1 : num_lim_mem_steps
                header = [header, strcat("f_a_m_", int2str(lim_mem_steps(i_lim_mem)))];
            end
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            data = [alpha, f, f_sw, f_u, f_rand, f_a, f_lim_mem];
            dlmwrite(filename, data, '-append');
        end
        
        % Wrtie fariness vs. k_avg to csv file
        function write_fairness_k_avg_csv(k_avg, ne_UF, fileprefix)
            % Make sure we have column vectors
            k_avg = reshape(k_avg, [], 1);
            f = -reshape(ne_UF, [], 1);
            
            % Header
            header = ["k_avg", "f"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            data = [k_avg, f];
            dlmwrite(filename, data, '-append');
        end
        
        % Sets axis limit 'scale' above/below min-max values
        function axis_semi_tight(ax, scale)
            axis tight; % Set axis tight
            % x limits
            xl = xlim(ax); % Get tight axis limits
            range = xl(2) - xl(1); % Get tight axis range
            sc_range = scale * range; % Scale range
            xl(1) = xl(1) - (sc_range - range) / 2; % New xmin
            xl(2) = xl(1) + sc_range; % New xmax
            xlim(ax, xl);
            % y limits
            yl = ylim(ax); % Get tight axis limits
            range = yl(2) - yl(1); % Get tight axis range
            sc_range = scale * range; % Scale range
            yl(1) = yl(1) - (sc_range - range) / 2; % New ymin
            yl(2) = yl(1) + sc_range; % New ymax
            ylim(ax, yl);
        end

        % Sets y-axis limit 'scale' above/below min-max values
        function y_semi_tight(ax, scale)
            axis tight; % Set axis tight
            yl = ylim(ax); % Get tight axis limits
            range = yl(2) - yl(1); % Get tight axis range
            sc_range = scale * range; % Scale range
            yl(1) = yl(1) - (sc_range - range) / 2; % New ymin
            yl(2) = yl(1) + sc_range; % New ymax
            ylim(ax, yl);
        end
    end
end