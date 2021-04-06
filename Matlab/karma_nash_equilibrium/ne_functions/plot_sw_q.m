% Plot SW expected stage reward
function plot_sw_q(fg, position, sw_q_down_u_k, U, K, k_ave)
    persistent sw_q_plot
    num_U = length(U);
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        sw_q_plot = cell(num_U, 1);
        lgd_text = cell(num_U, 1);
        sw_q_plot{1} = plot(K, -sw_q_down_u_k(1,:), '-x', 'LineWidth', 2);
        lgd_text{1} = ['u = ', num2str(U(1))];
        hold on;
        for i_u = 2 : num_U
            sw_q_plot{i_u} = plot(K, -sw_q_down_u_k(i_u,:), '-x', 'LineWidth', 2);
            lgd_text{i_u} = ['u = ', num2str(U(i_u))];
        end
        axis tight;
        axes = gca;
        axes.Title.FontName = 'ubuntu';
        axes.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW expected stage reward'];
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
    else
        for i_u = 1 : num_U
            sw_q_plot{i_u}.YData = -sw_q_down_u_k(i_u,:);
        end
        sw_q_plot{1}.Parent.Title.String = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW expected stage reward'];
    end
end