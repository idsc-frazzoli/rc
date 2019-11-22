classdef func
    % Functions used in scripts
    methods(Static)
        % Returns all maximizers (if there are multiple)
        function [max_v, max_i] = multi_maxes(input)
            [max_v, max_i] = max(input);
            input(max_i) = -realmax;
            [next_max_v, next_max_i] = max(input);
            while next_max_v == max_v
                max_i = [max_i, next_max_i];
                input(next_max_i) = -realmax;
                [next_max_v, next_max_i] = max(input);
            end
        end

        % Checks if there are multiple maximizers and returns one uniformly at
        % random
        function [max_v, max_i] = multi_max(input)
            % Get maximizers
            [max_v, max_i] = func.multi_maxes(input);
            % Choose one uniformly at random if there are multiple
            num_max = length(max_i);
            if num_max > 1
                max_i = max_i(ceil(rand(1) * num_max));
            end
        end

        % Gets delayed agents given index of passing agent
        function d = get_d(I, p_i)    
            d = I;
            d(p_i) = [];
        end

        % Gets karma paid by passing agent to delayed agents
        function [k_p, k_d] = get_karma_payments(m_p, d, curr_k, param)
            % Distribute karma evenly over delayed agents. If an agent will max
            % out their karma, tough luck!
            k_p_per_d = floor(m_p / param.num_d);
            k_d = zeros(1, param.num_d);
            for i = 1 : param.num_d
                k_d(i) = min([k_p_per_d, param.k_max - curr_k(d(i))]);
            end
            % Sum back the total karma distributed, which takes into account
            % delayed agents for which karma will saturate. This is the final
            % total paid by passing agent
            k_p = sum(k_d);
        end

        % Gets accumulated costs counting in warm-up period reset
        function a = get_accumulated_cost(c, param)
            a = [cumsum(c(1:param.t_warm_up,:)); cumsum(c(param.t_warm_up+1:end,:))];
        end

        % Standardizes input distribution given mean and variance vectors
        function output = standardize(input, input_mean, input_var)
            output = (input - input_mean) ./ sqrt(input_var);
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