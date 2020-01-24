clear;
close all;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
default_width = screenwidth / 2;
default_height = screenheight / 2;
load('karma_nash_equilibrium/RedColormap.mat');
fg = 1;

%% Some parameters
k_max = 12;
k_ave_vec = 0 : 12;
alpha_vec = 0 : 0.05 : 1;

num_k_ave = length(k_ave_vec);
num_alpha = length(alpha_vec);

ne_dir_str = ['karma_nash_equilibrium/results/k_max_', num2str(k_max, '%2d'), '_k_ave_'];
sw_dir = ['karma_nash_equilibrium/results/sw_k_max_', num2str(k_max, '%2d')];

%% Efficiency arrays
ne_e = zeros(num_k_ave, num_alpha);
sw_e = zeros(num_k_ave, 1);

for i_k_ave = 1 : num_k_ave
    k_ave = k_ave_vec(i_k_ave);
    % NE efficiencies
    ne_dir = [ne_dir_str, num2str(k_ave, '%02d')];
    for i_alpha = 1 : num_alpha
        alpha = alpha_vec(i_alpha);
        ne_file = [ne_dir, '/alpha_', num2str(alpha, '%.2f'), '.mat'];
        load(ne_file, 'ne_d_up_u_k', 'ne_q_down_u_k');
        ne_e(i_k_ave,i_alpha) = dot(reshape(ne_d_up_u_k, [], 1), reshape(ne_q_down_u_k, [], 1));
    end
    % SW efficiencies
    sw_file = [sw_dir, '/k_ave_', num2str(k_ave, '%02d'), '.mat'];
    load(sw_file, 'sw_d_up_u_k', 'sw_q_down_u_k');
    sw_e(i_k_ave) = dot(reshape(sw_d_up_u_k, [], 1), reshape(sw_q_down_u_k, [], 1));
end

%% Plot
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
lgd_text = cell(num_alpha + 1, 1);
plot(k_ave_vec, ne_e(:,1), '-x', 'LineWidth', 2);
lgd_text{1} = ['\alpha = ', num2str(alpha_vec(1), '%.2f')];
hold on;
for i_alpha = 2 : num_alpha
    plot(k_ave_vec, ne_e(:,i_alpha), '-x', 'LineWidth', 2);
    lgd_text{i_alpha} = ['\alpha = ', num2str(alpha_vec(i_alpha), '%.2f')];
end
plot(k_ave_vec, sw_e, '-o', 'LineWidth', 4);
lgd_text{end} = 'SW policy';
axis tight;
axes = gca;
axes.Title.FontName = 'ubuntu';
axes.Title.String = 'Efficiency of different policies';
axes.Title.FontSize = 12;
axes.XAxis.FontSize = 10;
axes.YAxis.FontSize = 10;
axes.XLabel.FontName = 'ubuntu';
axes.XLabel.String = 'Average karma';
axes.XLabel.FontSize = 12;
axes.YLabel.FontName = 'ubuntu';
axes.YLabel.String = 'Efficiency';
axes.YLabel.FontSize = 12;
lgd = legend(lgd_text);
lgd.FontSize = 12;
lgd.FontName = 'ubuntu';
lgd.Location = 'bestoutside';

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
    plot(alpha_vec, ne_e(i_k_ave,:), '-x', 'LineWidth', 2);
    lgd_text{1} = 'NE';
    hold on;
    plot(alpha_vec, repmat(sw_e(i_k_ave), 1, num_alpha), '-', 'LineWidth', 4);
    lgd_text{2} = 'SW';
    axis tight;
    axes = gca;
    axes.Title.FontName = 'ubuntu';
    axes.Title.String = ['k_{ave} = ', num2str(k_ave_vec(i_k_ave), '%02d'), ' efficiency'];
    axes.Title.FontSize = 12;
    axes.XAxis.FontSize = 10;
    axes.YAxis.FontSize = 10;
    axes.XLabel.FontName = 'ubuntu';
    axes.XLabel.String = 'Future discount factor';
    axes.XLabel.FontSize = 12;
    axes.YLabel.FontName = 'ubuntu';
    axes.YLabel.String = 'Efficiency';
    axes.YLabel.FontSize = 12;
%     lgd = legend(lgd_text);
%     lgd.FontSize = 12;
%     lgd.FontName = 'ubuntu';
%     lgd.Location = 'bestoutside';
end