% Plot NE payoffs
function plot_ne_J(fg, position, ne_J_down_mu_alpha_u_k, param, ne_param, i_alpha_comp)
    persistent ne_J_plot
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        ne_J_plot = cell(param.n_mu, param.n_alpha, param.n_u);
        n_subplots = param.n_mu * param.n_alpha;
        n_cols = ceil(sqrt(n_subplots));
        n_rows = ceil(n_subplots / n_cols);
        lgd_text = cell(param.n_u, 1);
        for i_u = 1 : param.n_u
            lgd_text{i_u} = ['u = ', num2str(param.U(i_u))];
        end
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                i_subplot = (i_mu - 1) * param.n_alpha + i_alpha;
                subplot(n_rows, n_cols, i_subplot);
                hold on;
                for i_u = 1 : param.n_u
                    ne_J_plot{i_mu,i_alpha,i_u} = plot(ne_param.K, squeeze(-ne_J_down_mu_alpha_u_k(i_mu,i_alpha,i_u,:)), '-x', 'LineWidth', 2);
                end
                axis tight;
                axes = gca;
                alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                if alpha == 1
                    ylim(axes, [mean(-ne_J_down_mu_alpha_u_k(:))*1.2, mean(-ne_J_down_mu_alpha_u_k(:))*0.8]);
                end
                axes.Title.FontName = 'ubuntu';
                if alpha > 0.99 && alpha < 1
                    alpha_str = num2str(alpha, '%.3f');
                else
                    alpha_str = num2str(alpha, '%.2f');
                end
                axes.Title.String = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE payoffs for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str];
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
            end
        end
    else
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                for i_u = 1 : param.n_u
                    ne_J_plot{i_mu,i_alpha,i_u}.YData = squeeze(-ne_J_down_mu_alpha_u_k(i_mu,i_alpha,i_u,:));
                end
                alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                if alpha == 1
                    ylim(ne_J_plot{i_mu,i_alpha,1}.Parent, [mean(-ne_J_down_mu_alpha_u_k(:))*1.2, mean(-ne_J_down_mu_alpha_u_k(:))*0.8]);
                end
                if alpha > 0.99 && alpha < 1
                    alpha_str = num2str(alpha, '%.3f');
                else
                    alpha_str = num2str(alpha, '%.2f');
                end
                ne_J_plot{i_mu,i_alpha,1}.Parent.Title.String = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE payoffs for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str];
            end
        end
    end
end