% Plot karma distribution
function plot_karma_dist(fg, position, k_dist, K, k_bar, alpha)
    figure(fg);
    fig = gcf;
    fig.Position = position;
    k_dist_mean = mean(k_dist);
    k_dist_std = std(k_dist, 1, 1);
    bar(K, k_dist_mean);
    hold on;
%             errorbar(K, k_dist_mean, k_dist_std, '--', 'LineWidth', 2);
    axis tight;
    axes = gca;
    axes.Title.FontName = 'ubuntu';
    axes.Title.String = ['\bar{k} = ', num2str(k_bar, '%02d'), ' \alpha = ', num2str(alpha, '%.2f'), ' karma distribution'];
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