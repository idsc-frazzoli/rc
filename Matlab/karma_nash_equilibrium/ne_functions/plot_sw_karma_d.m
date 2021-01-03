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