classdef ne_func
    % Miscellaneous helper functions used in scripts
    methods(Static)
        % Gets stationary distribution of transition matrix T
        % This essentially solves C = T.'*D. The solution is the left eigenvector
        % corresponding to eigenvalue 1, or the kernel of (I - T.')
        function D = stat_dist(T)
            n = size(T, 1);
            left_eig_T_1 = null(eye(n) - T.');
            % Make sure to return a valid probability distribution (sums to 1)
            if ~isempty(left_eig_T_1)
                D = left_eig_T_1 / sum(left_eig_T_1);
            else
                D = 1 / n * ones(n, 1);
            end
        end
        
        % Plot NE policy
        function plot_ne_pi(fg, position, colormap, ne_pi_down_u_k_up_m, U, K, M, alpha)
            persistent ne_pi_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_pi_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    pi_mat = squeeze(ne_pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat == 0) = nan;
                    subplot(1, num_U, i_u);
                    ne_pi_plot{i_u} = heatmap(K, M, pi_mat.', 'ColorbarVisible','off');
                    ne_pi_plot{i_u}.YDisplayData = flipud(ne_pi_plot{i_u}.YDisplayData);
                    ne_pi_plot{i_u}.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE policy for u = ', num2str(U(i_u))];
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
                    pi_mat(pi_mat == 0) = nan;
                    ne_pi_plot{i_u}.ColorData = pi_mat.';
                    ne_pi_plot{i_u}.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE policy for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot best response policy
        function plot_br_pi(fg, position, colormap, br_pi_down_u_k_up_m, U, K, M, alpha)
            persistent br_pi_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                br_pi_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    pi_mat = squeeze(br_pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat == 0) = nan;
                    subplot(1, num_U, i_u);
                    br_pi_plot{i_u} = heatmap(K, M, pi_mat.', 'ColorbarVisible','off');
                    br_pi_plot{i_u}.YDisplayData = flipud(br_pi_plot{i_u}.YDisplayData);
                    br_pi_plot{i_u}.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' BR policy for u = ', num2str(U(i_u))];
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
                    pi_mat(pi_mat == 0) = nan;
                    br_pi_plot{i_u}.ColorData = pi_mat.';
                    br_pi_plot{i_u}.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' BR policy for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE stationary distribution
        function plot_ne_d(fg, position, ne_d_up_u_k, U, K, alpha)
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
                    axes.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE stationary distribution for u = ', num2str(U(i_u))];
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
                    ne_d_plot{i_u}.Parent.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE stationary distribution for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE expected utility        
        function plot_ne_v(fg, position, ne_v_down_u_k, U, K, alpha)
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
                axes.Title.FontName = 'ubuntu';
                axes.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE expected utility'];
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
                ne_v_plot{1}.Parent.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE expected utility'];
            end
        end
        
        % Plot NE expected utility per message
        function plot_ne_rho(fg, position, colormap, ne_rho_down_u_k_m, U, K, M, alpha)
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
                    ne_rho_plot{i_u}.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE expected utility per message for u = ', num2str(U(i_u))];
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
                    ne_rho_plot{i_u}.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE expected utility per message for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE state transitions
        function plot_ne_t(fg, position, colormap, ne_t_down_u_k_up_un_kn, U, K, alpha)
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
                ne_t_plot.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE state transitions'];
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
                ne_t_plot.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE state transitions'];
            end
        end
        
        % Plot NE policy error
        function plot_ne_pi_error(fg, position, ne_pi_error_hist, alpha)
            persistent ne_pi_error_plot
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_pi_error_plot = plot(ne_pi_error_hist, 'r-x', 'LineWidth', 2);
                axis tight;
                axes = gca;
                axes.Title.FontName = 'ubuntu';
                axes.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE policy error'];
                axes.Title.FontSize = 12;
                axes.XAxis.FontSize = 10;
                axes.YAxis.FontSize = 10;
                axes.XLabel.FontName = 'ubuntu';
                axes.XLabel.String = 'Iteration';
                axes.XLabel.FontSize = 12;
                axes.YLabel.FontName = 'ubuntu';
                axes.YLabel.String = 'RMS policy error';
                axes.YLabel.FontSize = 12;
            else
                ne_pi_error_plot.YData = ne_pi_error_hist;
                ne_pi_error_plot.Parent.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE policy error'];
            end
        end
    end
end