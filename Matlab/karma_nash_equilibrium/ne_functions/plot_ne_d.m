% Plot NE population distribution
function plot_ne_d(fg, position, ne_d_up_mu_alpha_u_k, param, ne_param, i_alpha_comp)
    persistent ne_d_plot
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        ne_d_plot = cell(param.n_mu, param.n_alpha, param.n_u);
        n_subplots = param.n_mu * param.n_alpha * param.n_u;
        n_cols = ceil(sqrt(n_subplots));
        n_rows = ceil(n_subplots / n_cols);
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                for i_u = 1 : param.n_u
                    i_subplot = (i_mu - 1) * (param.n_alpha * param.n_u) + (i_alpha - 1) * param.n_u + i_u;
                    subplot(n_rows, n_cols, i_subplot);
                    ne_d_plot{i_mu,i_alpha,i_u} = bar(ne_param.K, squeeze(ne_d_up_mu_alpha_u_k(i_mu,i_alpha,i_u,:)));
                    axis tight;
                    axes = gca;
                    axes.Title.FontName = 'ubuntu';
                    alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                    if alpha > 0.99 && alpha < 1
                        alpha_str = num2str(alpha, '%.3f');
                    else
                        alpha_str = num2str(alpha, '%.2f');
                    end
                    axes.Title.String = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE distribution for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str, ', u = ', num2str(param.U(i_u))];
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
            end
        end
    else
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                for i_u = 1 : param.n_u
                    ne_d_plot{i_mu,i_alpha,i_u}.YData = ne_d_up_mu_alpha_u_k(i_mu,i_alpha,i_u,:);
                    alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                    if alpha > 0.99 && alpha < 1
                        alpha_str = num2str(alpha, '%.3f');
                    else
                        alpha_str = num2str(alpha, '%.2f');
                    end
                    ne_d_plot{i_mu,i_alpha,i_u}.Parent.Title.String = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE distribution for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str, ', u = ', num2str(param.U(i_u))];
                end
            end
        end
    end
end