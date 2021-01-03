% Plot NE policy error
function plot_ne_pi_error(fg, position, ne_pi_error_hist, param, i_alpha_comp)
    persistent ne_pi_error_plot
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        ne_pi_error_plot = plot(ne_pi_error_hist, 'r-x', 'LineWidth', 2);
        axis tight;
        axes = gca;
        axes.Title.FontName = 'ubuntu';
        title_str = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE policy error'];
        if param.n_alpha == 1
            alpha = param.Alpha(i_alpha_comp);
            if alpha > 0.99 && alpha < 1
                alpha_str = num2str(alpha, '%.3f');
            else
                alpha_str = num2str(alpha, '%.2f');
            end
            title_str = [title_str, ' for \alpha = ', alpha_str];
        end
        axes.Title.String = title_str;
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
        title_str = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' NE policy error'];
        if param.n_alpha == 1
            alpha = param.Alpha(i_alpha_comp);
            if alpha > 0.99 && alpha < 1
                alpha_str = num2str(alpha, '%.3f');
            else
                alpha_str = num2str(alpha, '%.2f');
            end
            title_str = [title_str, ' for \alpha = ', alpha_str];
        end
        ne_pi_error_plot.Parent.Title.String = title_str;
    end
end