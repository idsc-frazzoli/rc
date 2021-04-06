% Plot SW stationary distribution
function plot_sw_d(fg, position, sw_d_up_u_k, U, K, k_ave)
    persistent sw_d_plot
    num_U = length(U);
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        sw_d_plot = cell(num_U, 1);
        for i_u = 1 : num_U
            subplot(1, num_U, i_u);
            sw_d_plot{i_u} = bar(K, sw_d_up_u_k(i_u,:));
            axis tight;
            axes = gca;
            axes.Title.FontName = 'ubuntu';
            axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW stationary distribution for u = ', num2str(U(i_u))];
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
            sw_d_plot{i_u}.YData = sw_d_up_u_k(i_u,:);
            sw_d_plot{i_u}.Parent.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW stationary distribution for u = ', num2str(U(i_u))];
        end
    end
end