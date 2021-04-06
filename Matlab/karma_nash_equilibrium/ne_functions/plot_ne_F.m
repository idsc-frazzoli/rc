% Plot NE payoffs per bid
function plot_ne_F(fg, position, colormap, ne_F_down_mu_alpha_u_k_b, param, ne_param, i_alpha_comp)
    persistent ne_F_plot
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        ne_F_plot = cell(param.n_mu, param.n_alpha, param.n_u);
        n_subplots = param.n_mu * param.n_alpha * param.n_u;
        n_cols = ceil(sqrt(n_subplots));
        n_rows = ceil(n_subplots / n_cols);
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                for i_u = 1 : param.n_u
                    F_mat = squeeze(ne_F_down_mu_alpha_u_k_b(i_mu,i_alpha,i_u,:,:));
                    for i_k = 1 : ne_param.n_k
                        F_mat(i_k,ne_param.K>ne_param.K(i_k)) = nan;
                    end
                    i_subplot = (i_mu - 1) * (param.n_alpha * param.n_u) + (i_alpha - 1) * param.n_u + i_u;
                    subplot(n_rows, n_cols, i_subplot);
                    ne_F_plot{i_mu,i_alpha,i_u} = heatmap(ne_param.K, ne_param.K, -F_mat.', 'ColorbarVisible','off');
                    ne_F_plot{i_mu,i_alpha,i_u}.YDisplayData = flipud(ne_F_plot{i_mu,i_alpha,i_u}.YDisplayData);
                    alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                    if alpha > 0.99 && alpha < 1
                        alpha_str = num2str(alpha, '%.3f');
                    else
                        alpha_str = num2str(alpha, '%.2f');
                    end
                    ne_F_plot{i_mu,i_alpha,i_u}.Title = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE payoffs per bid for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str, ', u = ', num2str(param.U(i_u))];
                    ne_F_plot{i_mu,i_alpha,i_u}.XLabel = 'Karma';
                    ne_F_plot{i_mu,i_alpha,i_u}.YLabel = 'Bid';
                    ne_F_plot{i_mu,i_alpha,i_u}.FontName = 'Ubuntu';
                    ne_F_plot{i_mu,i_alpha,i_u}.FontSize = 10;
                    if exist('colormap', 'var')
                        ne_F_plot{i_mu,i_alpha,i_u}.Colormap = colormap;
                    end
                    ne_F_plot{i_mu,i_alpha,i_u}.CellLabelFormat = '%.2f';
                end
            end
        end
    else
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                for i_u = 1 : param.n_u
                    F_mat = squeeze(ne_F_down_mu_alpha_u_k_b(i_mu,i_alpha,i_u,:,:));
                    for i_k = 1 : ne_param.n_k
                        F_mat(i_k,ne_param.K>ne_param.K(i_k)) = nan;
                    end
                    ne_F_plot{i_mu,i_alpha,i_u}.ColorData = -F_mat.';
                    alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                    if alpha > 0.99 && alpha < 1
                        alpha_str = num2str(alpha, '%.3f');
                    else
                        alpha_str = num2str(alpha, '%.2f');
                    end
                    ne_F_plot{i_mu,i_alpha,i_u}.Title = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE payoffs per bid for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str, ', u = ', num2str(param.U(i_u))];
                end
            end
        end
    end
end