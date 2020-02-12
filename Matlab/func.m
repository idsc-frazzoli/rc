classdef func
    % Functions used in scripts
    methods(Static)
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
        function s_up_k_uniform = get_s_up_k_uniform(param)
            i_kave = find(param.K == param.k_ave);
            if param.k_ave * 2 <= param.k_max
                i_kave2 = find(param.K == param.k_ave * 2);
                s_up_k_uniform = [1 / i_kave2 * ones(i_kave2, 1); zeros(param.num_K - i_kave2, 1)];
            elseif param.k_ave >= param.k_max
                s_up_k_uniform = zeros(param.num_K, 1);
                s_up_k_uniform(end) = 1;
            else
                s_up_k_uniform = 1 / param.num_K * ones(param.num_K, 1);
                K_small = 0 : param.k_ave - 1;
                K_big = param.k_ave + 1 : param.k_max;
                num_K_small = length(K_small);
                num_K_big = length(K_big);
                delta_constant = sum(K_small) / num_K_small - sum(K_big) / num_K_big;
                delta_k_ave = param.k_ave - param.K.' * s_up_k_uniform;
                delta_p = delta_k_ave / delta_constant;
                s_up_k_uniform(1:i_kave-1) = s_up_k_uniform(1:i_kave-1) + delta_p / num_K_small;
                s_up_k_uniform(i_kave+1:end) = s_up_k_uniform(i_kave+1:end) - delta_p / num_K_big;
            end
        end
        
        % Gets uniform distribution over karma without karma saturation for
        % infinite population N
        function [s_up_k_uniform, K_uniform] = get_s_up_k_uniform_dynamic(k_ave)
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
            
            % Shuffle initial karma in the end because why not
            init_k = init_k(randperm(param.N));
        end
        
        % Gets agents that lose given indeces of agents that win
        function lose = get_lose(I, win_i)
            lose = I;
            lose(win_i) = [];
        end

        % Gets karma paid by winning agent to losing agents
        function [k_win, k_lose] = get_karma_payments(m_win, lose, k, param)
            % Distribute karma evenly over losing agents. If an agent will
            % max out their karma, tough luck!
            k_win_per_lose = floor(m_win / param.num_lose);
            k_lose = zeros(1, param.num_lose);
            for i_lose = 1 : param.num_lose
                k_lose(i_lose) = min([k_win_per_lose, param.k_max - k(lose(i_lose))]);
            end
            % Sum back the total karma distributed, which takes into
            % account losing agents for which karma will saturate. This is
            % the final total paid by winning agent
            k_win = sum(k_lose);
        end
        
        % Gets karma paid by winning agent to losing agents
        function [k_win, k_lose] = get_karma_payments_dynamic(m_win, param)
            % Distribute karma evenly over losing agents, flooring
            k_win_per_lose = floor(m_win / param.num_lose);
            k_lose = k_win_per_lose * ones(1, param.num_lose);
            % Sum back the total karma distributed, which takes into
            % account flooring effect. This is the final total paid by
            % winning agent
            k_win = sum(k_lose);
        end

        % Gets accumulated costs counting in warm-up period reset
        function a = get_accumulated_cost(c, param)
            a = [cumsum(c(1:param.t_warm_up,:)); cumsum(c(param.t_warm_up+1:end,:))];
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