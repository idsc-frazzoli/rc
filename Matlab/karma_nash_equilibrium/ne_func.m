classdef ne_func
    % Miscellaneous helper functions used in scripts
    methods(Static)
        % Gets game tensors
        function [c_down_u_m_mj, kappa_down_k_m_up_kn_down_mj] = get_game_tensors(ne_param)
            % Probability of winning/losing given messages
            gamma_down_m_mj_up_o = zeros(ne_param.num_K, ne_param.num_K, ne_param.num_O);
            for i_m = 1 : ne_param.num_K
                m = ne_param.K(i_m);
                for i_mj = 1 : ne_param.num_K
                    mj = ne_param.K(i_mj);
                    gamma_down_m_mj_up_o(i_m,i_mj,1) = max([0, min([(m - mj + 1) / 2, 1])]);
                    gamma_down_m_mj_up_o(i_m,i_mj,2) = 1 - gamma_down_m_mj_up_o(i_m,i_mj,1);
                end
            end

            % Game cost tensor
            zeta_down_u_o = outer(ne_param.U, ne_param.O);
            c_down_u_m_mj = squeeze(dot2(zeta_down_u_o, permute(gamma_down_m_mj_up_o, [3 1 2]), 2, 1));
            clearvars zeta_down_u_o;

            % Game state transition tensor
            phi_down_k_m_mj_o_up_kn = ...
                zeros(ne_param.num_K, ne_param.num_K, ne_param.num_K, ne_param.num_O, ne_param.num_K);
            for i_k = 1 : ne_param.num_K
                k = ne_param.K(i_k);
                for i_m = 1 : i_k
                    m = ne_param.K(i_m);
                    for i_mj = 1 : ne_param.num_K
                        mj = ne_param.K(i_mj);

                        switch ne_param.m_exchange
                            case 0      % Pay as bid
                                i_kn_win = find(ne_param.K == k - m);
                                i_kn_lose = find(ne_param.K == min([k + mj, ne_param.k_max]));
                            case 1      % Pay difference
                                i_kn_win = find(ne_param.K == k - (m - mj));
                                i_kn_lose = find(ne_param.K == min([k + (mj - m), ne_param.k_max]));
                            case 2      % Pay difference and pay one on tie
                                if m == mj && m >= 1
                                    i_kn_win = find(ne_param.K == k - 1);
                                    i_kn_lose = find(ne_param.K == min([k + 1, ne_param.k_max]));
                                else
                                    i_kn_win = find(ne_param.K == k - (m - mj));
                                    i_kn_lose = find(ne_param.K == min([k + (mj - m), ne_param.k_max]));
                                end
                        end

                        if ~isempty(i_kn_win)
                            phi_down_k_m_mj_o_up_kn(i_k,i_m,i_mj,1,i_kn_win) = 1;
                        end  
                        if ~isempty(i_kn_lose)
                            phi_down_k_m_mj_o_up_kn(i_k,i_m,i_mj,2,i_kn_lose) = 1;
                        end
                    end
                end
            end
            kappa_down_k_m_up_kn_down_mj = permute(dot2(permute(phi_down_k_m_mj_o_up_kn, [1 5 2 3 4]), gamma_down_m_mj_up_o, 5, 3), [1 3 2 4]);
            clearvars gamma_down_m_mj_up_o phi_down_k_m_mj_o_up_kn;
        end
        
        % Gets initial policy guess
        function pi_down_u_k_up_m_init = get_pi_init(ne_param)
            pi_down_u_k_up_m_init = zeros(ne_param.num_U, ne_param.num_K, ne_param.num_K);
            switch ne_param.pi_init_method
                case 0 
                    % Bid urgency
                    for i_u = 1 : ne_param.num_U
                        for i_k = 1 : ne_param.num_K
                            i_m = min([i_u, i_k]);
                            pi_down_u_k_up_m_init(i_u,i_k,i_m) = 1;
                        end
                    end
                case 1
                    % Bid 0.5 * u / u_max * k (~ 'bid half if urgent')
                    for i_u = 1 : ne_param.num_U
                        for i_k = 1 : ne_param.num_K
                            m = round(0.5 * ne_param.U(i_u) / ne_param.U(end) * ne_param.K(i_k));
                            i_m = ne_param.K == m;
                            pi_down_u_k_up_m_init(i_u,i_k,i_m) = 1;
                        end
                    end
                case 2
                    % Bid 1 * u / u_max * k (~ 'bid all if urgent')
                    for i_u = 1 : ne_param.num_U
                        for i_k = 1 : ne_param.num_K
                            m = round(ne_param.U(i_u) / ne_param.U(end) * ne_param.K(i_k));
                            i_m = ne_param.K == m;
                            pi_down_u_k_up_m_init(i_u,i_k,i_m) = 1;
                        end
                    end
                case 3
                    % Bid random
                    for i_u = 1 : ne_param.num_U
                        for i_k = 1 : ne_param.num_K
                            pi_down_u_k_up_m_init(i_u,i_k,1:i_k) = 1 / i_k;
                        end
                    end
            end
        end
        
        % Gets initial distribution guess
        function [p_up_u_init, s_up_k_init, d_up_u_k_init] = get_d_init(k_ave, ne_param)
            p_up_u_init = ne_param.p_up_u;
            s_up_k_init = ne_func.get_s_up_k_uniform(k_ave, ne_param);
            d_up_u_k_init = outer(p_up_u_init, s_up_k_init);
        end
        
        % Gets uniform distribution over karma adjusted to have correct
        % average karma k_ave
        function s_up_k_uniform = get_s_up_k_uniform(k_ave, param)
            i_kave = find(param.K == k_ave);
            if k_ave * 2 <= param.k_max
                i_kave2 = find(param.K == k_ave * 2);
                s_up_k_uniform = [1 / i_kave2 * ones(i_kave2, 1); zeros(param.num_K - i_kave2, 1)];
            elseif k_ave >= param.k_max
                s_up_k_uniform = zeros(param.num_K, 1);
                s_up_k_uniform(end) = 1;
            else
                s_up_k_uniform = 1 / param.num_K * ones(param.num_K, 1);
                K_small = 0 : k_ave - 1;
                K_big = k_ave + 1 : param.k_max;
                num_K_small = length(K_small);
                num_K_big = length(K_big);
                delta_constant = sum(K_small) / num_K_small - sum(K_big) / num_K_big;
                delta_k_ave = k_ave - param.K.' * s_up_k_uniform;
                delta_p = delta_k_ave / delta_constant;
                s_up_k_uniform(1:i_kave-1) = s_up_k_uniform(1:i_kave-1) + delta_p / num_K_small;
                s_up_k_uniform(i_kave+1:end) = s_up_k_uniform(i_kave+1:end) - delta_p / num_K_big;
            end
        end
        
        % Finds the median of the distribution
        function med = find_median(s_up_k, K)
            s_up_k = s_up_k / sum(s_up_k);
            s_up_k_cumsum = cumsum(s_up_k);
            i_med = find(s_up_k_cumsum >= 0.5);
            med = K(i_med(1));
        end
        
        % Computes the efficiency of an interaction coset-stationary
        % distribution pair
        function e = compute_efficiency(q_down_u_k, d_up_u_k)
            e = -dot(reshape(d_up_u_k, [], 1), reshape(q_down_u_k, [], 1));
        end
        
        % Computes the optimal efficiency (which corresponds to centralized
        % urgency)
        function e_opt = optimal_efficiency(ne_param)
            nu_up_u = func.stat_dist(ne_param.mu_down_u_up_un);
            e_opt = 0;
            for i_ui = 1 : ne_param.num_U
                e_opt = e_opt - 0.5 * nu_up_u(i_ui)^2 * ne_param.U(i_ui);
                for i_uj = i_ui + 1 : ne_param.num_U
                    e_opt = e_opt - nu_up_u(i_ui) * nu_up_u(i_uj) * ne_param.U(i_ui);
                end
            end
        end
        
        % Efficiency of baseline random
        function e_rand = random_efficiency(ne_param)
            nu_up_u = func.stat_dist(ne_param.mu_down_u_up_un);
            e_rand = -0.5 * dot(nu_up_u, ne_param.U);
        end
        
        % Infinite horizon payoff of baseline random
        function J_rand = random_inf_horizon_payoff(ne_param, alpha)
            if alpha == 1
                J_rand = ne_func.random_efficiency(ne_param) * ones(ne_param.num_U, 1);
            else
                Q = -0.5 * ne_param.U;
                J_rand = (eye(ne_param.num_U) - alpha * ne_param.mu_down_u_up_un) \ Q;
            end
        end
        
        % Plot NE policy
        function plot_ne_pi(fg, position, colormap, ne_pi_down_u_k_up_m, U, K, M, k_ave, alpha)
            persistent ne_pi_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_pi_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    pi_mat = squeeze(ne_pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat <= eps) = nan;
                    subplot(1, num_U, i_u);
                    ne_pi_plot{i_u} = heatmap(K, M, pi_mat.', 'ColorbarVisible','off');
                    ne_pi_plot{i_u}.YDisplayData = flipud(ne_pi_plot{i_u}.YDisplayData);
                    ne_pi_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE policy for u = ', num2str(U(i_u))];
                    ne_pi_plot{i_u}.XLabel = 'Karma';
                    ne_pi_plot{i_u}.YLabel = 'Message';
                    ne_pi_plot{i_u}.FontName = 'Ubuntu';
                    ne_pi_plot{i_u}.FontSize = 10;
                    if exist('colormap', 'var')
                        ne_pi_plot{i_u}.Colormap = colormap;
                    end
                    ne_pi_plot{i_u}.ColorLimits = [0 1];
                    ne_pi_plot{i_u}.CellLabelFormat = '%.2f';
                end
            else
                for i_u = 1 : num_U
                    pi_mat = squeeze(ne_pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat <= eps) = nan;
                    ne_pi_plot{i_u}.ColorData = pi_mat.';
                    ne_pi_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE policy for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot best response policy
        function plot_br_pi(fg, position, colormap, br_pi_down_u_k_up_m, U, K, M, k_ave, alpha)
            persistent br_pi_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                br_pi_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    pi_mat = squeeze(br_pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat <= eps) = nan;
                    subplot(1, num_U, i_u);
                    br_pi_plot{i_u} = heatmap(K, M, pi_mat.', 'ColorbarVisible','off');
                    br_pi_plot{i_u}.YDisplayData = flipud(br_pi_plot{i_u}.YDisplayData);
                    br_pi_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' BR policy for u = ', num2str(U(i_u))];
                    br_pi_plot{i_u}.XLabel = 'Karma';
                    br_pi_plot{i_u}.YLabel = 'Message';
                    br_pi_plot{i_u}.FontName = 'Ubuntu';
                    br_pi_plot{i_u}.FontSize = 10;
                    if exist('colormap', 'var')
                        br_pi_plot{i_u}.Colormap = colormap;
                    end
                    br_pi_plot{i_u}.ColorLimits = [0 1];
                    br_pi_plot{i_u}.CellLabelFormat = '%.2f';
                end
            else
                for i_u = 1 : num_U
                    pi_mat = squeeze(br_pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat <= eps) = nan;
                    br_pi_plot{i_u}.ColorData = pi_mat.';
                    br_pi_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' BR policy for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot social welfare policy
        function plot_sw_pi(fg, position, colormap, sw_pi_down_u_k_up_m, U, K, M, k_ave)
            persistent sw_pi_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                sw_pi_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    pi_mat = squeeze(sw_pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat <= 1e-6) = nan;
                    subplot(1, num_U, i_u);
                    sw_pi_plot{i_u} = heatmap(K, M, pi_mat.', 'ColorbarVisible','off');
                    sw_pi_plot{i_u}.YDisplayData = flipud(sw_pi_plot{i_u}.YDisplayData);
                    sw_pi_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW policy for u = ', num2str(U(i_u))];
                    sw_pi_plot{i_u}.XLabel = 'Karma';
                    sw_pi_plot{i_u}.YLabel = 'Message';
                    sw_pi_plot{i_u}.FontName = 'Ubuntu';
                    sw_pi_plot{i_u}.FontSize = 10;
                    if exist('colormap', 'var')
                        sw_pi_plot{i_u}.Colormap = colormap;
                    end
                    sw_pi_plot{i_u}.ColorLimits = [0 1];
                    sw_pi_plot{i_u}.CellLabelFormat = '%.2f';
                end
            else
                for i_u = 1 : num_U
                    pi_mat = squeeze(sw_pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat <= 1e-6) = nan;
                    sw_pi_plot{i_u}.ColorData = pi_mat.';
                    sw_pi_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW policy for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE stationary distribution
        function plot_ne_d(fg, position, ne_d_up_u_k, U, K, k_ave, alpha)
            persistent ne_d_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_d_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    subplot(1, num_U, i_u);
                    ne_d_plot{i_u} = bar(K, ne_d_up_u_k(i_u,:));
                    axis tight;
                    axes = gca;
                    axes.Title.FontName = 'ubuntu';
                    axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE stationary distribution for u = ', num2str(U(i_u))];
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
            else
                for i_u = 1 : num_U
                    ne_d_plot{i_u}.YData = ne_d_up_u_k(i_u,:);
                    ne_d_plot{i_u}.Parent.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE stationary distribution for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE stationary karma distribution
        function plot_ne_karma_d(fg, position, ne_s_up_k, K, k_ave, alpha)
        persistent ne_karma_d_plot
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_karma_d_plot = bar(K, ne_s_up_k);
                axis tight;
                axes = gca;
                axes.Title.FontName = 'ubuntu';
                axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE stationary karma distribution'];
                axes.Title.FontSize = 12;
                axes.XAxis.FontSize = 10;
                axes.YAxis.FontSize = 10;
                axes.XLabel.FontName = 'ubuntu';
                axes.XLabel.String = 'Karma';
                axes.XLabel.FontSize = 12;
                axes.YLabel.FontName = 'ubuntu';
                axes.YLabel.String = 'Probability';
                axes.YLabel.FontSize = 12;
            else
                ne_karma_d_plot.YData = ne_s_up_k;
                ne_karma_d_plot.Parent.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE stationary karma distribution'];
            end
        end
        
        % Plot SW stationary distribution
        function plot_sw_d(fg, position, sw_d_up_u_k, U, K, k_ave)
            persistent sw_d_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                sw_d_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    subplot(1, num_U, i_u);
                    sw_d_plot{i_u} = bar(K, sw_d_up_u_k(i_u,:));
                    axis tight;
                    axes = gca;
                    axes.Title.FontName = 'ubuntu';
                    axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW stationary distribution for u = ', num2str(U(i_u))];
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
            else
                for i_u = 1 : num_U
                    sw_d_plot{i_u}.YData = sw_d_up_u_k(i_u,:);
                    sw_d_plot{i_u}.Parent.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW stationary distribution for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot SW stationary karma distribution
        function plot_sw_karma_d(fg, position, sw_s_up_k, K, k_ave)
            persistent sw_karma_d_plot
                if ~ishandle(fg)
                    figure(fg);
                    fig = gcf;
                    fig.Position = position;
                    sw_karma_d_plot = bar(K, sw_s_up_k);
                    axis tight;
                    axes = gca;
                    axes.Title.FontName = 'ubuntu';
                    axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW stationary karma distribution'];
                    axes.Title.FontSize = 12;
                    axes.XAxis.FontSize = 10;
                    axes.YAxis.FontSize = 10;
                    axes.XLabel.FontName = 'ubuntu';
                    axes.XLabel.String = 'Karma';
                    axes.XLabel.FontSize = 12;
                    axes.YLabel.FontName = 'ubuntu';
                    axes.YLabel.String = 'Probability';
                    axes.YLabel.FontSize = 12;
                else
                    sw_karma_d_plot.YData = sw_s_up_k;
                    sw_karma_d_plot.Parent.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW stationary karma distribution'];
                end
        end
        
        % Plot NE expected utility        
        function plot_ne_v(fg, position, ne_v_down_u_k, U, K, k_ave, alpha)
            persistent ne_v_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_v_plot = cell(num_U, 1);
                lgd_text = cell(num_U, 1);
                ne_v_plot{1} = plot(K, -ne_v_down_u_k(1,:), '-x', 'LineWidth', 2);
                lgd_text{1} = ['u = ', num2str(U(1))];
                hold on;
                for i_u = 2 : num_U
                    ne_v_plot{i_u} = plot(K, -ne_v_down_u_k(i_u,:), '-x', 'LineWidth', 2);
                    lgd_text{i_u} = ['u = ', num2str(U(i_u))];
                end
                axis tight;
                axes = gca;
                if alpha == 1
                    ylim(axes, [mean(-ne_v_down_u_k(:))*1.2, mean(-ne_v_down_u_k(:))*0.8]);
                end
                axes.Title.FontName = 'ubuntu';
                axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE expected utility'];
                axes.Title.FontSize = 12;
                axes.XAxis.FontSize = 10;
                axes.YAxis.FontSize = 10;
                axes.XLabel.FontName = 'ubuntu';
                axes.XLabel.String = 'Karma';
                axes.XLabel.FontSize = 12;
                axes.YLabel.FontName = 'ubuntu';
                axes.YLabel.String = 'Utility';
                axes.YLabel.FontSize = 12;
                lgd = legend(lgd_text);
                lgd.FontSize = 12;
                lgd.FontName = 'ubuntu';
                lgd.Location = 'bestoutside';
            else
                for i_u = 1 : num_U
                    ne_v_plot{i_u}.YData = -ne_v_down_u_k(i_u,:);
                end
                ne_v_plot{1}.Parent.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE expected utility'];
            end
        end
        
        % Plot SW expected stage reward
        function plot_sw_q(fg, position, sw_q_down_u_k, U, K, k_ave)
            persistent sw_q_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                sw_q_plot = cell(num_U, 1);
                lgd_text = cell(num_U, 1);
                sw_q_plot{1} = plot(K, -sw_q_down_u_k(1,:), '-x', 'LineWidth', 2);
                lgd_text{1} = ['u = ', num2str(U(1))];
                hold on;
                for i_u = 2 : num_U
                    sw_q_plot{i_u} = plot(K, -sw_q_down_u_k(i_u,:), '-x', 'LineWidth', 2);
                    lgd_text{i_u} = ['u = ', num2str(U(i_u))];
                end
                axis tight;
                axes = gca;
                axes.Title.FontName = 'ubuntu';
                axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW expected stage reward'];
                axes.Title.FontSize = 12;
                axes.XAxis.FontSize = 10;
                axes.YAxis.FontSize = 10;
                axes.XLabel.FontName = 'ubuntu';
                axes.XLabel.String = 'Karma';
                axes.XLabel.FontSize = 12;
                axes.YLabel.FontName = 'ubuntu';
                axes.YLabel.String = 'Utility';
                axes.YLabel.FontSize = 12;
                lgd = legend(lgd_text);
                lgd.FontSize = 12;
                lgd.FontName = 'ubuntu';
                lgd.Location = 'bestoutside';
            else
                for i_u = 1 : num_U
                    sw_q_plot{i_u}.YData = -sw_q_down_u_k(i_u,:);
                end
                sw_q_plot{1}.Parent.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW expected stage reward'];
            end
        end
        
        % Plot NE expected utility per message
        function plot_ne_rho(fg, position, colormap, ne_rho_down_u_k_m, U, K, M, k_ave, alpha)
            persistent ne_rho_plot
            num_U = length(U);
            num_K = length(K);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_rho_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    rho_mat = squeeze(ne_rho_down_u_k_m(i_u,:,:));
                    for i_k = 1 : num_K
                        rho_mat(i_k,M>K(i_k)) = nan;
                    end
                    subplot(1, num_U, i_u);
                    ne_rho_plot{i_u} = heatmap(K, M, -rho_mat.', 'ColorbarVisible','off');
                    ne_rho_plot{i_u}.YDisplayData = flipud(ne_rho_plot{i_u}.YDisplayData);
                    ne_rho_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE expected utility per message for u = ', num2str(U(i_u))];
                    ne_rho_plot{i_u}.XLabel = 'Karma';
                    ne_rho_plot{i_u}.YLabel = 'Message';
                    ne_rho_plot{i_u}.FontName = 'Ubuntu';
                    ne_rho_plot{i_u}.FontSize = 10;
                    if exist('colormap', 'var')
                        ne_rho_plot{i_u}.Colormap = colormap;
                    end
                    ne_rho_plot{i_u}.CellLabelFormat = '%.2f';
                end
            else
                for i_u = 1 : num_U
                    rho_mat = squeeze(ne_rho_down_u_k_m(i_u,:,:));
                    for i_k = 1 : num_K
                        rho_mat(i_k,M>K(i_k)) = nan;
                    end
                    ne_rho_plot{i_u}.ColorData = -rho_mat.';
                    ne_rho_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE expected utility per message for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE state transitions
        function plot_ne_t(fg, position, colormap, ne_t_down_u_k_up_un_kn, U, K, k_ave, alpha)
            persistent ne_t_plot
            num_U = length(U);
            num_K = length(K);
            num_X = num_U * num_K;
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                label = cell(num_X, 1);
                for i_u = 1 : num_U
                    base_i_u = (i_u - 1) * num_K;
                    u_str = num2str(U(i_u));
                    for i_k = 1 : num_K
                        label{base_i_u+i_k} = ['(', u_str, ',', num2str(K(i_k)), ')'];
                    end
                end
                t_mat = zeros(num_X);
                for i_u = 1 : num_U
                    start_i_u = (i_u - 1) * num_K + 1;
                    end_i_u = i_u * num_K;
                    for i_un = 1 : num_U
                        start_i_un = (i_un - 1) * num_K + 1;
                        end_i_un = i_un * num_K;
                        t_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                            squeeze(ne_t_down_u_k_up_un_kn(i_u,:,i_un,:));
                    end
                end
                ne_t_plot = heatmap(label, label, t_mat.', 'ColorbarVisible','off');
                ne_t_plot.YDisplayData = flipud(ne_t_plot.YDisplayData);
                ne_t_plot.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE state transitions'];
                ne_t_plot.XLabel = 'State now (urgency,karma)';
                ne_t_plot.YLabel = 'State next (urgency,karma)';
                ne_t_plot.FontName = 'Ubuntu';
                ne_t_plot.FontSize = 10;
                if exist('colormap', 'var')
                    ne_t_plot.Colormap = colormap;
                end
                ne_t_plot.CellLabelFormat = '%.2f';
            else
                t_mat = zeros(num_X);
                for i_u = 1 : num_U
                    start_i_u = (i_u - 1) * num_K + 1;
                    end_i_u = i_u * num_K;
                    for i_un = 1 : num_U
                        start_i_un = (i_un - 1) * num_K + 1;
                        end_i_un = i_un * num_K;
                        t_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                            squeeze(ne_t_down_u_k_up_un_kn(i_u,:,i_un,:));
                    end
                end
                ne_t_plot.ColorData = t_mat.';
                ne_t_plot.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE state transitions'];
            end
        end
        
        % Plot SW state transitions
        function plot_sw_t(fg, position, colormap, sw_t_down_u_k_up_un_kn, U, K, k_ave)
            persistent sw_t_plot
            num_U = length(U);
            num_K = length(K);
            num_X = num_U * num_K;
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                label = cell(num_X, 1);
                for i_u = 1 : num_U
                    base_i_u = (i_u - 1) * num_K;
                    u_str = num2str(U(i_u));
                    for i_k = 1 : num_K
                        label{base_i_u+i_k} = ['(', u_str, ',', num2str(K(i_k)), ')'];
                    end
                end
                t_mat = zeros(num_X);
                for i_u = 1 : num_U
                    start_i_u = (i_u - 1) * num_K + 1;
                    end_i_u = i_u * num_K;
                    for i_un = 1 : num_U
                        start_i_un = (i_un - 1) * num_K + 1;
                        end_i_un = i_un * num_K;
                        t_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                            squeeze(sw_t_down_u_k_up_un_kn(i_u,:,i_un,:));
                    end
                end
                sw_t_plot = heatmap(label, label, t_mat.', 'ColorbarVisible','off');
                sw_t_plot.YDisplayData = flipud(sw_t_plot.YDisplayData);
                sw_t_plot.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW state transitions'];
                sw_t_plot.XLabel = 'State now (urgency,karma)';
                sw_t_plot.YLabel = 'State next (urgency,karma)';
                sw_t_plot.FontName = 'Ubuntu';
                sw_t_plot.FontSize = 10;
                if exist('colormap', 'var')
                    sw_t_plot.Colormap = colormap;
                end
                sw_t_plot.CellLabelFormat = '%.2f';
            else
                t_mat = zeros(num_X);
                for i_u = 1 : num_U
                    start_i_u = (i_u - 1) * num_K + 1;
                    end_i_u = i_u * num_K;
                    for i_un = 1 : num_U
                        start_i_un = (i_un - 1) * num_K + 1;
                        end_i_un = i_un * num_K;
                        t_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                            squeeze(sw_t_down_u_k_up_un_kn(i_u,:,i_un,:));
                    end
                end
                sw_t_plot.ColorData = t_mat.';
                sw_t_plot.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW state transitions'];
            end
        end
        
        % Plot NE policy error
        function plot_ne_pi_error(fg, position, ne_pi_error_hist, k_ave, alpha)
            persistent ne_pi_error_plot
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_pi_error_plot = plot(ne_pi_error_hist, 'r-x', 'LineWidth', 2);
                axis tight;
                axes = gca;
                axes.Title.FontName = 'ubuntu';
                axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE policy error'];
                axes.Title.FontSize = 12;
                axes.XAxis.FontSize = 10;
                axes.YAxis.FontSize = 10;
                axes.XLabel.FontName = 'ubuntu';
                axes.XLabel.String = 'Iteration';
                axes.XLabel.FontSize = 12;
                axes.YLabel.FontName = 'ubuntu';
                axes.YLabel.String = 'Policy error';
                axes.YLabel.FontSize = 12;
            else
                ne_pi_error_plot.YData = ne_pi_error_hist;
                ne_pi_error_plot.Parent.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' NE policy error'];
            end
        end
        
        % Wrtie policy to csv file
        function num_K = write_pi_csv(pi_down_u_k_up_m, s_up_k, U, K, pi_tol, s_tol, fileprefix)
            num_U = length(U);
            num_K = length(K);
            
            % Remove tail of distribution where there are too few agents
            while s_up_k(num_K) < s_tol
                num_K = num_K - 1;
            end
            
            % Remove 'zero' values
            pi_down_u_k_up_m(pi_down_u_k_up_m < pi_tol) = 0;
            for i_u = 1 : num_U
                for i_k = 1 : num_K
                    pi_down_u_k_up_m(i_u,i_k,1:num_K) = pi_down_u_k_up_m(i_u,i_k,1:num_K) / sum(pi_down_u_k_up_m(i_u,i_k,1:num_K));
                end
            end
            
            % Header
            header = ["u", "k", "k2", "b", "b2", "P(b)"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Header for mean of policy
            header_mean = ["u", "k", "b"];
            filename_mean = [fileprefix, '_mean.csv'];
            fout = fopen(filename_mean, 'w');
            for i = 1 : length(header_mean) - 1
                fprintf(fout, '%s,', header_mean(i));
            end
            fprintf(fout, '%s\n', header_mean(end));
            fclose(fout);
            
            % Data
            for i_u = 1 : num_U
                u = U(i_u);
                for i_k = 1 : num_K + 1
                    k = i_k - 1;
                    for i_b = 1 : num_K + 1
                        b = i_b - 1;
                        if i_b <= i_k && i_k <= num_K
                            line = [u, k, k - 0.5, b, b - 0.5, pi_down_u_k_up_m(i_u,i_k,i_b)];
                        else
                            line = [u, k, k - 0.5, b, b - 0.5, 2];
                        end
                        dlmwrite(filename, line, '-append');
                    end
                    
                    if i_k <= num_K
                        line_mean = [U(i_u), K(i_k), dot(squeeze(pi_down_u_k_up_m(i_u,i_k,:)), K)];
                        dlmwrite(filename_mean, line_mean, '-append');
                    end
                end
            end
        end
        
        % Wrtie stationary karma distribution to csv file
        function num_K = write_s_csv(s_up_k, K, s_tol, fileprefix)
            num_K = length(K);
            
            % Remove tail of distribution where there are too few agents
            while s_up_k(num_K) < s_tol
                num_K = num_K - 1;
            end
            if num_K == length(K)
                K = [K; K(end)+1];
                s_up_k = [s_up_k; 0];
            end
            
            % Renormalize
            s_up_k = s_up_k / sum(s_up_k);
            
            % Header
            header = ["k", "k2", "P(k)"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            data = [K(1:num_K+1), K(1:num_K+1) - 0.5, s_up_k(1:num_K+1)];
            dlmwrite(filename, data, '-append');
        end
        
        % Wrtie stationary distribution to csv file
        function num_K = write_d_csv(d_up_u_k, U, K, s_tol, fileprefix)
            num_U = length(U);
            num_K = length(K);
            s_up_k = sum(d_up_u_k).';
            
            % Remove tail of distribution where there are too few agents
            while s_up_k(num_K) < s_tol
                num_K = num_K - 1;
            end
            if num_K == length(K)
                K = [K; K(end)+1];
                d_up_u_k = [d_up_u_k, zeros(num_U, 1)];
            end
            
            % Header
            header = ["u", "k", "k2", "P(k)"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            for i_u = 1 : num_U
                data = [U(i_u) * ones(num_K + 1, 1), K(1:num_K+1), K(1:num_K+1) - 0.5, d_up_u_k(i_u,1:num_K+1).'];
                dlmwrite(filename, data, '-append');
            end
        end
        
        % Wrtie infinite horizon payoff to csv file
        function num_K = write_J_csv(v_down_u_k, s_up_k, ne_param, alpha, s_tol, fileprefix)
            num_K = ne_param.num_K;
            
            % Remove tail of distribution where there are too few agents
            while s_up_k(num_K) < s_tol
                num_K = num_K - 1;
            end
            
            % Header
            header = ["u", "k", "J"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Header for baseline random payoffs
            filename_rand = [fileprefix, '_rand.csv'];
            fout = fopen(filename_rand, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Compute infinite horizon payoffs for baseline random
            J_down_u_rand = ne_func.random_inf_horizon_payoff(ne_param, alpha);
            
            % Data
            for i_u = 1 : ne_param.num_U
                for i_k = 1 : num_K
                    line = [ne_param.U(i_u), ne_param.K(i_k), -v_down_u_k(i_u,i_k)];
                    dlmwrite(filename, line, '-append');
                    
                    line_rand = [ne_param.U(i_u), ne_param.K(i_k), J_down_u_rand(i_u)];
                    dlmwrite(filename_rand, line_rand, '-append');
                end
            end
        end
        
        % Wrtie transition matrix to csv file
        function num_K = write_t_csv(t_down_u_k_up_un_kn, s_up_k, U, K, s_tol, fileprefix)
            num_U = length(U);
            num_K = length(K);
            
            % Remove tail of distribution where there are too few agents
            while s_up_k(num_K) < s_tol
                num_K = num_K - 1;
            end
            
            % Header
            header = ["u", "k", "k2", "kn", "kn2", "T"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            
            % Data
            for i_u = 1 : num_U
                u = U(i_u);
                for i_k = 1 : num_K + 1
                    k = i_k - 1;
                    for i_kn = 1 : num_K + 1
                        kn = i_kn - 1;
                        line = [u, k, k - 0.5, kn, kn - 0.5, sum(t_down_u_k_up_un_kn(i_u,i_k,:,i_kn))];
                        dlmwrite(filename, line, '-append');
                    end
                end
            end
        end
        
        % Wrtie karma stationary distribution to csv file
        function num_K = write_iota_csv(pi_down_u_k_up_m, d_up_u_k, K, iota_tol, fileprefix)
            num_K = length(K);
            
            % Compute the probability distribution of the messages
            iota_up_m = dot2(reshape(permute(pi_down_u_k_up_m, [3 1 2]), num_K, []), reshape(d_up_u_k, [], 1), 2, 1);
            
            % Remove tail of policy where there are too few agents
            while iota_up_m(num_K) < iota_tol
                num_K = num_K - 1;
            end
            
            % Renormalize
            iota_up_m = iota_up_m / sum(iota_up_m);
            
            % Header
            header = ["b", "b2", "P(b)"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            data = [K(1:num_K+1), K(1:num_K+1) - 0.5, iota_up_m(1:num_K+1)];
            dlmwrite(filename, data, '-append');
        end
        
        % Wrtie price of karma vs. alpha to csv file
        function write_PoK_alpha_csv(alpha_vec, e, e_opt, e_rand, fileprefix)
            % Make sure we have column vectors
            alpha_vec = reshape(alpha_vec, [], 1);
            e = reshape(e, [], 1);
            
            % Add efficiency for alpha = 0, which is baseline random
            % efficiency
            e(alpha_vec == 0) = [];
            alpha_vec(alpha_vec == 0) = [];
            alpha_vec = [0; alpha_vec];
            e = [e_rand; e];
            
            % Make vectors out of e_opt and e_rand for plotting
            num_Alpha = length(alpha_vec);
            e_opt = e_opt * ones(num_Alpha, 1);
            e_rand = e_rand * ones(num_Alpha, 1);
            
            % PoK
            PoK = e ./ e_opt;
            PoK_opt = e_opt ./ e_opt;
            PoK_rand = e_rand ./ e_opt;
            
            % Header
            header = ["alpha", "e", "PoK", "e_opt", "PoK_opt", "e_rand", "PoK_rand"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            data = [alpha_vec, e, PoK, e_opt, PoK_opt, e_rand, PoK_rand];
            dlmwrite(filename, data, '-append');
        end
        
        % Wrtie price of karma vs. k_avg to csv file
        function write_PoK_k_avg_csv(k_avg_vec, e, e_opt, e_rand, fileprefix)
            % Make sure we have column vectors
            k_avg_vec = reshape(k_avg_vec, [], 1);
            e = reshape(e, [], 1);
            
            % Add efficiency for alpha = 0, which is baseline random
            % efficiency
            e(k_avg_vec == 0) = [];
            k_avg_vec(k_avg_vec == 0) = [];
            k_avg_vec = [0; k_avg_vec];
            e = [e_rand; e];
            
            % Make vectors out of e_opt and e_rand for plotting
            num_k_avg = length(k_avg_vec);
            e_opt = e_opt * ones(num_k_avg, 1);
            e_rand = e_rand * ones(num_k_avg, 1);
            
            % PoK
            PoK = e ./ e_opt;
            PoK_opt = e_opt ./ e_opt;
            PoK_rand = e_rand ./ e_opt;
            
            % Header
            header = ["k_avg", "e", "PoK", "e_opt", "PoK_opt", "e_rand", "PoK_rand"];
            filename = [fileprefix, '.csv'];
            fout = fopen(filename, 'w');
            for i = 1 : length(header) - 1
                fprintf(fout, '%s,', header(i));
            end
            fprintf(fout, '%s\n', header(end));
            fclose(fout);
            
            % Data
            data = [k_avg_vec, e, PoK, e_opt, PoK_opt, e_rand, PoK_rand];
            dlmwrite(filename, data, '-append');
        end
    end
end