classdef results_func
    % Miscellaneous helper functions used in scripts
    methods(Static)
        % Plot NE policy
        function plot_ne_pi(fg, position, colormap, pi_down_u_k_up_m, U, K, alpha)
            persistent ne_pi_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_pi_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    pi_mat = squeeze(pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat == 0) = nan;
                    subplot(1, num_U, i_u);
                    ne_pi_plot{i_u} = heatmap(K, K, pi_mat.', 'ColorbarVisible','off');
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
                    pi_mat = squeeze(pi_down_u_k_up_m(i_u,:,:));
                    pi_mat(pi_mat == 0) = nan;
                    ne_pi_plot{i_u}.ColorData = pi_mat.';
                    ne_pi_plot{i_u}.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE policy for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE stationary distribution
        function plot_ne_D(fg, position, D_up_u_k, U, K, alpha)
            persistent ne_D_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_D_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    subplot(1, num_U, i_u);
                    ne_D_plot{i_u} = bar(K, D_up_u_k(i_u,:));
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
                    ne_D_plot{i_u}.YData = D_up_u_k(i_u,:);
                    ne_D_plot{i_u}.Parent.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE stationary distribution for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE expected utility
        function plot_ne_V(fg, position, V_down_u_k, U, K, alpha)
            persistent ne_V_plot
            num_U = length(U);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_V_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    subplot(1, num_U, i_u);
                    ne_V_plot{i_u} = plot(K, -V_down_u_k(i_u,:), '-x', 'LineWidth', 2);
                    axis tight;
                    axes = gca;
                    axes.Title.FontName = 'ubuntu';
                    axes.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE expected utility for u = ', num2str(U(i_u))];
                    axes.Title.FontSize = 12;
                    axes.XAxis.FontSize = 10;
                    axes.YAxis.FontSize = 10;
                    axes.XLabel.FontName = 'ubuntu';
                    axes.XLabel.String = 'Karma';
                    axes.XLabel.FontSize = 12;
                    axes.YLabel.FontName = 'ubuntu';
                    axes.YLabel.String = 'Utility';
                    axes.YLabel.FontSize = 12;
                end
            else
                for i_u = 1 : num_U
                    ne_V_plot{i_u}.YData = -V_down_u_k(i_u,:);
                    ne_V_plot{i_u}.Parent.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE expected utility for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE expected utility per message
        function plot_ne_V_m(fg, position, colormap, V_down_u_k_m, U, K, alpha)
            persistent ne_V_m_plot
            num_U = length(U);
            num_K = length(K);
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_V_m_plot = cell(num_U, 1);
                for i_u = 1 : num_U
                    V_m_mat = squeeze(V_down_u_k_m(i_u,:,:));
                    for i_k = 1 : num_K
                        V_m_mat(i_k,i_k+1:end) = nan;
                    end
                    subplot(1, num_U, i_u);
                    ne_V_m_plot{i_u} = heatmap(K, K, -V_m_mat.', 'ColorbarVisible','off');
                    ne_V_m_plot{i_u}.YDisplayData = flipud(ne_V_m_plot{i_u}.YDisplayData);
                    ne_V_m_plot{i_u}.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE expected utility per message for u = ', num2str(U(i_u))];
                    ne_V_m_plot{i_u}.XLabel = 'Karma';
                    ne_V_m_plot{i_u}.YLabel = 'Message';
                    ne_V_m_plot{i_u}.FontName = 'Ubuntu';
                    ne_V_m_plot{i_u}.FontSize = 10;
                    if exist('colormap', 'var')
                        ne_V_m_plot{i_u}.Colormap = colormap;
                    end
                    ne_V_m_plot{i_u}.CellLabelFormat = '%.2f';
                end
            else
                for i_u = 1 : num_U
                    V_m_mat = squeeze(V_down_u_k_m(i_u,:,:));
                    for i_k = 1 : num_K
                        V_m_mat(i_k,i_k+1:end) = nan;
                    end
                    ne_V_m_plot{i_u}.ColorData = -V_m_mat.';
                    ne_V_m_plot{i_u}.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE expected utility per message for u = ', num2str(U(i_u))];
                end
            end
        end
        
        % Plot NE state transitions
        function plot_ne_T(fg, position, colormap, T_down_u_k_up_un_kn, U, K, alpha)
            persistent ne_T_plot
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
                T_mat = zeros(num_X);
                for i_u = 1 : num_U
                    start_i_u = (i_u - 1) * num_K + 1;
                    end_i_u = i_u * num_K;
                    for i_un = 1 : num_U
                        start_i_un = (i_un - 1) * num_K + 1;
                        end_i_un = i_un * num_K;
                        T_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                            squeeze(T_down_u_k_up_un_kn(i_u,:,i_un,:));
                    end
                end
                ne_T_plot = heatmap(label, label, T_mat.', 'ColorbarVisible','off');
                ne_T_plot.YDisplayData = flipud(ne_T_plot.YDisplayData);
                ne_T_plot.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE state transitions'];
                ne_T_plot.XLabel = 'State now (urgency,karma)';
                ne_T_plot.YLabel = 'State next (urgency,karma)';
                ne_T_plot.FontName = 'Ubuntu';
                ne_T_plot.FontSize = 10;
                if exist('colormap', 'var')
                    ne_T_plot.Colormap = colormap;
                end
                ne_T_plot.CellLabelFormat = '%.2f';
            else
                T_mat = zeros(num_X);
                for i_u = 1 : num_U
                    start_i_u = (i_u - 1) * num_K + 1;
                    end_i_u = i_u * num_K;
                    for i_un = 1 : num_U
                        start_i_un = (i_un - 1) * num_K + 1;
                        end_i_un = i_un * num_K;
                        T_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                            squeeze(T_down_u_k_up_un_kn(i_u,:,i_un,:));
                    end
                end
                ne_T_plot.ColorData = T_mat.';
                ne_T_plot.Title = ['\alpha = ', num2str(alpha, '%.2f'), ' NE state transitions'];
            end
        end
        
        % Plot NE policy error
        function plot_ne_pi_error(fg, position, ne_policy_error_hist, alpha)
            persistent ne_pi_error_plot
            if ~ishandle(fg)
                figure(fg);
                fig = gcf;
                fig.Position = position;
                ne_pi_error_plot = plot(ne_policy_error_hist, 'r-x', 'LineWidth', 2);
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
                ne_pi_error_plot.YData = ne_policy_error_hist;
                ne_pi_error_plot.Parent.Title.String = ['\alpha = ', num2str(alpha, '%.2f'), ' NE policy error'];
            end
        end
    end
end