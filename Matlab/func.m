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

        % Gets agents that lose given indeces of agents that win
        function lose = get_lose(I, win_i)
            lose = I;
            lose(win_i) = [];
        end

        % Gets karma paid by winning agent to losing agents
        function [k_win, k_lose] = get_karma_payments(m_win, lose, k, param)
            % Distribute karma evenly over losing agents. If an agent will max
            % out their karma, tough luck!
            k_win_per_lose = floor(m_win / param.num_lose);
            k_lose = zeros(1, param.num_win);
            for i_win = 1 : param.num_win
                k_lose(i_win) = min([k_win_per_lose, param.k_max - k(lose(i_win))]);
            end
            % Sum back the total karma distributed, which takes into account
            % losing agents for which karma will saturate. This is the final
            % total paid by wining agent
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