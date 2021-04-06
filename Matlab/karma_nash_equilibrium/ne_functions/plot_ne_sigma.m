% Plot NE stationary karma distribution
function plot_ne_sigma(fg, position, ne_sigma_down_mu_alpha_up_k, param, ne_param, i_alpha_comp)
persistent ne_sigma_plot
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        ne_sigma_plot = cell(param.n_mu, param.n_alpha);
        n_subplots = param.n_mu * param.n_alpha;
        n_cols = ceil(sqrt(n_subplots));
        n_rows = ceil(n_subplots / n_cols);
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                i_subplot = (i_mu - 1) * param.n_alpha + i_alpha;
                subplot(n_rows, n_cols, i_subplot);
                ne_sigma_plot{i_mu,i_alpha} = bar(ne_param.K, squeeze(ne_sigma_down_mu_alpha_up_k(i_mu,i_alpha,:)));
                axis tight;
                axes = gca;
                axes.Title.FontName = 'ubuntu';
                alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                if alpha > 0.99 && alpha < 1
                    alpha_str = num2str(alpha, '%.3f');
                else
                    alpha_str = num2str(alpha, '%.2f');
                end
                axes.Title.String = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE karma distribution for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str];
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
    else
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                ne_sigma_plot{i_mu,i_alpha}.YData = squeeze(ne_sigma_down_mu_alpha_up_k(i_mu,i_alpha,:));
                alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                if alpha > 0.99 && alpha < 1
                    alpha_str = num2str(alpha, '%.3f');
                else
                    alpha_str = num2str(alpha, '%.2f');
                end
                ne_sigma_plot{i_mu,i_alpha}.Parent.Title.String = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE karma distribution for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str];
            end
        end
    end
end