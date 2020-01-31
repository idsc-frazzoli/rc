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
k_max = 24;
k_ave_vec = 3 : 12;
alpha_vec = 0 : 0.05 : 1;

num_k_ave = length(k_ave_vec);
num_alpha = length(alpha_vec);

ne_dir_str = ['karma_nash_equilibrium/results/k_max_', num2str(k_max, '%2d'), '_k_ave_'];

sw_computed = false;
if sw_computed
    sw_dir = ['karma_nash_equilibrium/results/sw_k_max_', num2str(k_max, '%2d')];
end

%% Efficiency arrays
ne_W1 = zeros(num_k_ave, num_alpha);
if sw_computed
    sw_W1 = zeros(num_k_ave, 1);
end

for i_k_ave = 1 : num_k_ave
    k_ave = k_ave_vec(i_k_ave);
    % NE efficiencies
    ne_dir = [ne_dir_str, num2str(k_ave, '%02d')];
    for i_alpha = 1 : num_alpha
        alpha = alpha_vec(i_alpha);
        ne_file = [ne_dir, '/alpha_', num2str(alpha, '%.2f'), '.mat'];
        load(ne_file, 'ne_d_up_u_k', 'ne_q_down_u_k');
        ne_W1(i_k_ave,i_alpha) = dot(reshape(ne_d_up_u_k, [], 1), reshape(ne_q_down_u_k, [], 1));
    end
    % SW efficiencies
    if sw_computed
        sw_file = [sw_dir, '/k_ave_', num2str(k_ave, '%02d'), '.mat'];
        load(sw_file, 'sw_d_up_u_k', 'sw_q_down_u_k');
        sw_W1(i_k_ave) = dot(reshape(sw_d_up_u_k, [], 1), reshape(sw_q_down_u_k, [], 1));
    end
end

%% Plot all
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
if sw_computed
    lgd_text = cell(num_alpha + 1, 1);
else
    lgd_text = cell(num_alpha, 1);
end
plot(k_ave_vec, -ne_W1(:,1), '-x', 'LineWidth', 2);
lgd_text{1} = ['$\alpha$ = ', num2str(alpha_vec(1), '%.2f')];
hold on;
for i_alpha = 2 : num_alpha
    plot(k_ave_vec, -ne_W1(:,i_alpha), '-x', 'LineWidth', 2);
    lgd_text{i_alpha} = ['$\alpha$ = ', num2str(alpha_vec(i_alpha), '%.2f')];
end
if sw_computed
    plot(k_ave_vec, -sw_W1, '-o', 'LineWidth', 4);
    lgd_text{end} = 'SW policy';
end
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
    if sw_computed
        plot(alpha_vec, repmat(-sw_W1(i_k_ave), 1, num_alpha), '-', 'LineWidth', 4);
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
    if sw_computed
        plot(k_ave_vec, -sw_W1, '-', 'LineWidth', 4);
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
    axes.YLabel.String = 'Efficiency';
    axes.YLabel.FontSize = 12;
end
title = sgtitle('Efficiency as a function of $k_{ave}$ for different future discount factors');
title.Interpreter = 'latex';
title.FontSize = 16;