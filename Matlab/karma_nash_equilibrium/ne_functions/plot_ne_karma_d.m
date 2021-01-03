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