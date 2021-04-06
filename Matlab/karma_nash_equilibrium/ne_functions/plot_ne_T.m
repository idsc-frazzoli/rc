% Plot NE state transitions
function plot_ne_T(fg, position, colormap, ne_T_down_mu_alpha_u_k_up_un_kn, param, ne_param, i_alpha_comp)
    persistent ne_T_plot
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        ne_T_plot = cell(param.n_mu, param.n_alpha);
        n_subplots = param.n_mu * param.n_alpha;
        n_cols = ceil(sqrt(n_subplots));
        n_rows = ceil(n_subplots / n_cols);
        label = cell(ne_param.n_x, 1);
        for i_u = 1 : param.n_u
            base_i_u = (i_u - 1) * ne_param.n_k;
            u_str = num2str(param.U(i_u));
            for i_k = 1 : ne_param.n_k
                label{base_i_u+i_k} = ['(', u_str, ',', num2str(ne_param.K(i_k)), ')'];
            end
        end
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                i_subplot = (i_mu - 1) * param.n_alpha + i_alpha;
                subplot(n_rows, n_cols, i_subplot);
                T_mat = zeros(ne_param.n_x);
                for i_u = 1 : param.n_u
                    start_i_u = (i_u - 1) * ne_param.n_k + 1;
                    end_i_u = i_u * ne_param.n_k;
                    for i_un = 1 : param.n_u
                        start_i_un = (i_un - 1) * ne_param.n_k + 1;
                        end_i_un = i_un * ne_param.n_k;
                        T_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                            squeeze(ne_T_down_mu_alpha_u_k_up_un_kn(i_mu,i_alpha,i_u,:,i_un,:));
                    end
                end
                ne_T_plot{i_mu,i_alpha} = heatmap(label, label, T_mat.', 'ColorbarVisible','off');
                ne_T_plot{i_mu,i_alpha}.YDisplayData = flipud(ne_T_plot{i_mu,i_alpha}.YDisplayData);
                alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                if alpha > 0.99 && alpha < 1
                    alpha_str = num2str(alpha, '%.3f');
                else
                    alpha_str = num2str(alpha, '%.2f');
                end
                ne_T_plot{i_mu,i_alpha}.Title = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE state transitions for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str];
                ne_T_plot{i_mu,i_alpha}.XLabel = 'State now (urgency,karma)';
                ne_T_plot{i_mu,i_alpha}.YLabel = 'State next (urgency,karma)';
                ne_T_plot{i_mu,i_alpha}.FontName = 'Ubuntu';
                ne_T_plot{i_mu,i_alpha}.FontSize = 10;
                if exist('colormap', 'var')
                    ne_T_plot{i_mu,i_alpha}.Colormap = colormap;
                end
                ne_T_plot{i_mu,i_alpha}.CellLabelFormat = '%.2f';
            end
        end
    else
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                T_mat = zeros(ne_param.n_x);
                for i_u = 1 : param.n_u
                    start_i_u = (i_u - 1) * ne_param.n_k + 1;
                    end_i_u = i_u * ne_param.n_k;
                    for i_un = 1 : param.n_u
                        start_i_un = (i_un - 1) * ne_param.n_k + 1;
                        end_i_un = i_un * ne_param.n_k;
                        T_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                            squeeze(ne_T_down_mu_alpha_u_k_up_un_kn(i_mu,i_alpha,i_u,:,i_un,:));
                    end
                end
                ne_T_plot{i_mu,i_alpha}.ColorData = T_mat.';
                alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                if alpha > 0.99 && alpha < 1
                    alpha_str = num2str(alpha, '%.3f');
                else
                    alpha_str = num2str(alpha, '%.2f');
                end
                ne_T_plot{i_mu,i_alpha}.Title = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE state transitions for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str];
                ne_T_plot{i_mu,i_alpha}.XLabel = 'State now (urgency,karma)';
            end
        end
    end
end