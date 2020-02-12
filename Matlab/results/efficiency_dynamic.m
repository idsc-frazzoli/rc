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
k_ave_vec = 1 : 12;
alpha_vec = [0.05 : 0.05 : 0.95, 0.96 : 0.01 : 1];

num_k_ave = length(k_ave_vec);
num_alpha = length(alpha_vec);

file_str = 'results/dynamic/k_ave_';

%% Efficiency arrays
ne_W1 = zeros(num_k_ave, num_alpha);

for i_k_ave = 1 : num_k_ave
    file = [file_str, num2str(k_ave_vec(i_k_ave), '%02d'), '.mat'];
    % NE efficiencies
    load(file, 'W1_ne');
    for i_alpha = 1 : num_alpha
        ne_W1(i_k_ave,i_alpha) = W1_ne{i_alpha}(end);
    end
    % Centralized urgency efficiency (get once for comparison)
    if i_k_ave == 1
        load(file, 'W1_1');
        u_W1 = W1_1(end);
    end
end

%% Plot all
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
lgd_text = cell(num_alpha + 1, 1);
plot(k_ave_vec, -ne_W1(:,1), '-x', 'LineWidth', 2);
lgd_text{1} = ['$\alpha$ = ', num2str(alpha_vec(1), '%.2f')];
hold on;
for i_alpha = 2 : num_alpha
    plot(k_ave_vec, -ne_W1(:,i_alpha), '-x', 'LineWidth', 2);
    lgd_text{i_alpha} = ['$\alpha$ = ', num2str(alpha_vec(i_alpha), '%.2f')];
end
plot(k_ave_vec, repmat(-u_W1, 1, num_k_ave), 'r-', 'LineWidth', 4);
lgd_text{end} = 'centralized-urgency';
axis tight;
axes = gca;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'Efficiency of different policies';
axes.Title.FontSize = 16;
axes.XAxis.FontSize = 10;
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Average karma';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Efficiency';
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
    plot(alpha_vec, -ne_W1(i_k_ave,:), '-x', 'LineWidth', 2);
    hold on;
    plot(alpha_vec, repmat(-u_W1, 1, num_alpha), 'r-', 'LineWidth', 4);
    axis tight;
    ylim(yl);
    axes = gca;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = ['$k_{avg}$ = ', num2str(k_ave_vec(i_k_ave), '%02d')];
    axes.Title.FontSize = 14;
    axes.XAxis.FontSize = 10;
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Future discount factor';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Efficiency';
    axes.YLabel.FontSize = 12;
end
title = sgtitle('Efficiency as a function of $\alpha$ for different average karmas');
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
    plot(k_ave_vec, -ne_W1(:,i_alpha), '-x', 'LineWidth', 2);
    hold on;
    plot(k_ave_vec, repmat(-u_W1, 1, num_k_ave), 'r-', 'LineWidth', 4);
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
    axes.YLabel.String = 'Efficiency';
    axes.YLabel.FontSize = 12;
end
title = sgtitle('Efficiency as a function of $k_{avg}$ for different future discount factors');
title.Interpreter = 'latex';
title.FontSize = 16;