clear;
close all;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
default_width = screenwidth / 2;
default_height = screenheight / 2;
fg = 1;

%% Some parameters
k_max = 12;
k_ave_vec = 0 : 12;
alpha_vec = 0 : 0.05 : 1;

num_k_ave = length(k_ave_vec);
num_alpha = length(alpha_vec);

file_str = ['results/k_max_', num2str(k_max, '%2d'), '_k_ave_'];

sw_computed = true;
plot_u_c = false;

%% Fainress arrays
ne_W2 = zeros(num_k_ave, num_alpha);
if sw_computed
    sw_W2 = zeros(num_k_ave, 1);
end

for i_k_ave = 1 : num_k_ave
    file = [file_str, num2str(k_ave_vec(i_k_ave), '%02d'), '.mat'];
    % NE fairnesses
    load(file, 'W2_ne');
    for i_alpha = 1 : num_alpha
        ne_W2(i_k_ave,i_alpha) = W2_ne{i_alpha}(end);
    end
    % SW fairnesses
    if sw_computed
        load(file, 'W2_sw');
        sw_W2(i_k_ave) = W2_sw(end);
    end
    % Centralized urgency & urgency-then-cost fairness (get once for comparison)
    if i_k_ave == 1
        load(file, 'W2_1');
        u_W2 = W2_1(end);
        if plot_u_c
            load(file, 'W2_1_2');
            u_c_W2 = W2_1_2(end);
        end
    end
end

%% Plot all
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
num_plots = num_alpha + 1;
if sw_computed
    num_plots = num_plots + 1;
end
if plot_u_c
    num_plots = num_plots + 1;
end
lgd_text = cell(num_plots, 1);
plot(k_ave_vec, -ne_W2(:,1), '-x', 'LineWidth', 2);
lgd_text{1} = ['$\alpha$ = ', num2str(alpha_vec(1), '%.2f')];
hold on;
for i_plot = 2 : num_alpha
    plot(k_ave_vec, -ne_W2(:,i_plot), '-x', 'LineWidth', 2);
    lgd_text{i_plot} = ['$\alpha$ = ', num2str(alpha_vec(i_plot), '%.2f')];
end
i_plot = num_alpha + 1;
plot(k_ave_vec, repmat(-u_W2, 1, num_k_ave), '-', 'LineWidth', 4);
lgd_text{i_plot} = 'centralized-urgency';
if plot_u_c
    i_plot = i_plot + 1;
    plot(k_ave_vec, repmat(-u_c_W2, 1, num_k_ave), '-', 'LineWidth', 4);
    lgd_text{i_plot} = 'centralized-urgency-then-cost';
end
if sw_computed
    i_plot = i_plot + 1;
    plot(k_ave_vec, -sw_W2, '-o', 'LineWidth', 3);
    lgd_text{i_plot} = 'SW policy';
end
axis tight;
axes = gca;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'Fairness of different policies';
axes.Title.FontSize = 16;
axes.XAxis.FontSize = 10;
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Average karma';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Fairness';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.FontSize = 12;
lgd.Interpreter = 'latex';
lgd.Location = 'bestoutside';
yl = ylim;

%% Plot slices through k_ave
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
num_cols = round(sqrt(screenwidth / screenheight * num_k_ave));
num_rows = ceil(num_k_ave / num_cols);
for i_k_ave = 1 : num_k_ave
    subplot(num_rows,num_cols,i_k_ave);
    lgd_text = cell(2, 1);
    plot(alpha_vec, -ne_W2(i_k_ave,:), '-x', 'LineWidth', 2);
    hold on;
    plot(alpha_vec, repmat(-u_W2, 1, num_alpha), '-', 'LineWidth', 4);
    if plot_u_c
        plot(alpha_vec, repmat(-u_c_W2, 1, num_alpha), '-', 'LineWidth', 4);
    end
    if sw_computed
        plot(alpha_vec, repmat(-sw_W2(i_k_ave), 1, num_alpha), '-', 'LineWidth', 2);
    end
    axis tight;
    ylim(yl);
    axes = gca;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = ['$k_{ave}$ = ', num2str(k_ave_vec(i_k_ave), '%02d')];
    axes.Title.FontSize = 14;
    axes.XAxis.FontSize = 10;
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Future discount factor';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Fairness';
    axes.YLabel.FontSize = 12;
end
title = sgtitle('Fairness as a function of $\alpha$ for different average karmas');
title.Interpreter = 'latex';
title.FontSize = 16;

%% Plot slices through alpha
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
num_cols = round(sqrt(screenwidth / screenheight * num_alpha));
num_rows = ceil(num_alpha / num_cols);
for i_alpha = 1 : num_alpha
    subplot(num_rows,num_cols,i_alpha);
    lgd_text = cell(2, 1);
    plot(k_ave_vec, -ne_W2(:,i_alpha), '-x', 'LineWidth', 2);
    hold on;
    plot(k_ave_vec, repmat(-u_W2, 1, num_k_ave), '-', 'LineWidth', 4);
    if plot_u_c
        plot(k_ave_vec, repmat(-u_c_W2, 1, num_k_ave), '-', 'LineWidth', 4);
    end
    if sw_computed
        plot(k_ave_vec, -sw_W2, '-', 'LineWidth', 2);
    end
    axis tight;
    ylim(yl);
    axes = gca;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = ['$\alpha$ = ', num2str(alpha_vec(i_alpha), '%0.2f')];
    axes.Title.FontSize = 14;
    axes.XAxis.FontSize = 10;
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Average karma';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Fairness';
    axes.YLabel.FontSize = 12;
end
title = sgtitle('Fairness as a function of $k_{ave}$ for different future discount factors');
title.Interpreter = 'latex';
title.FontSize = 16;