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
K = 0 : k_max;
k_ave_vec = 1 : 11;
alpha_vec = 0 : 0.05 : 0.95;

num_k_ave = length(k_ave_vec);
num_alpha = length(alpha_vec);

dir_str = ['karma_nash_equilibrium/results/k_max_', num2str(k_max, '%2d'), '_k_ave_'];

%% k_max bid & p(k_max) arrays
k_max_bid = zeros(num_k_ave, num_alpha);
k_max_p = zeros(num_k_ave, num_alpha);

for i_k_ave = 1 : num_k_ave
    k_ave = k_ave_vec(i_k_ave);
    dir_pi = [dir_str, num2str(k_ave, '%02d'), '/policies/'];
    dir_D = [dir_str, num2str(k_ave, '%02d'), '/stationary_distributions/'];
    for i_alpha = 1 : num_alpha
        alpha = alpha_vec(i_alpha);
        file_pi = [dir_pi, 'alpha_', num2str(alpha, '%.2f'), '.mat'];
        file_D = [dir_D, 'alpha_', num2str(alpha, '%.2f'), '.mat'];
        load(file_pi);
        load(file_D);
        k_max_bid(i_k_ave,i_alpha) = round(K * squeeze(pi_down_u_k_up_m(2,end,:)) + 0.02, 1);
        k_max_p(i_k_ave,i_alpha) = sum(D_up_u_k(:,end));
    end
end

%% Find 'phase transistions'
% This is where agents start/stop bidding max karma at max level
transition_alpha = zeros(num_k_ave, 1);
for i_k_ave = 1 : num_k_ave
    k_max_bid_max_i = find(k_max_bid(i_k_ave,:) == k_max);
    transition_alpha(i_k_ave) = alpha_vec(k_max_bid_max_i(end));
end
transition_k_ave = zeros(num_alpha, 1);
for i_alpha = 1 : num_alpha
    k_max_bid_max_i = find(k_max_bid(:,i_alpha) == k_max);
    transition_k_ave(i_alpha) = k_ave_vec(k_max_bid_max_i(1));
end

%% Plot k_max_bid
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, default_width, default_height];
h = heatmap(k_ave_vec, alpha_vec, k_max_bid.', 'ColorbarVisible','off');
h.YDisplayData = flipud(h.YDisplayData);
h.Title = 'Average bid at maximum karma level';
h.XLabel = 'Average karma';
h.YLabel = 'Future discount factor';
h.FontName = 'Ubuntu';
h.FontSize = 10;
h.Colormap = parula;

%% Plot k_max_p
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [default_width, 0, default_width, default_height];
h = heatmap(k_ave_vec, alpha_vec, k_max_p.', 'ColorbarVisible','off');
h.YDisplayData = flipud(h.YDisplayData);
h.Title = 'Stationary probability of posessing maximum karma level';
h.XLabel = 'Average karma';
h.YLabel = 'Future discount factor';
h.FontName = 'Ubuntu';
h.FontSize = 10;
h.Colormap = RedColormap;
h.CellLabelFormat = '%.2f';

%% Plot slices for average karma
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, default_height, default_width, default_height];
lgd_text = cell(num_alpha, 1);
plot(k_ave_vec, k_max_bid(:,1), '-x', 'LineWidth', 2);
lgd_text{1} = ['\alpha = ', num2str(alpha_vec(1), '%.2f')];
hold on;
for i_alpha = 2 : num_alpha
    plot(k_ave_vec, k_max_bid(:,i_alpha), '-x', 'LineWidth', 2);
    lgd_text{i_alpha} = ['\alpha = ', num2str(alpha_vec(i_alpha), '%.2f')];
end
axis tight;
axes = gca;
axes.Title.FontName = 'ubuntu';
axes.Title.String = 'Average bid at maximum karma level for different future discount factors';
axes.Title.FontSize = 12;
axes.XAxis.FontSize = 10;
axes.YAxis.FontSize = 10;
axes.XLabel.FontName = 'ubuntu';
axes.XLabel.String = 'Average karma';
axes.XLabel.FontSize = 12;
axes.YLabel.FontName = 'ubuntu';
axes.YLabel.String = 'Average bid';
axes.YLabel.FontSize = 12;
lgd = legend(lgd_text);
lgd.FontSize = 12;
lgd.FontName = 'ubuntu';
lgd.Location = 'bestoutside';

%% Plot slices for alpha
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [default_width, default_height, default_width, default_height];
lgd_text = cell(num_k_ave, 1);
plot(alpha_vec, k_max_bid(1,:), '-x', 'LineWidth', 2);
lgd_text{1} = ['k_{ave} = ', num2str(k_ave_vec(1), '%02d')];
hold on;
for i_k_ave = 2 : num_k_ave
    plot(alpha_vec, k_max_bid(i_k_ave,:), '-x', 'LineWidth', 2);
    lgd_text{i_k_ave} = ['k_{ave} = ', num2str(k_ave_vec(i_k_ave), '%02d')];
end
axis tight;
axes = gca;
axes.Title.FontName = 'ubuntu';
axes.Title.String = 'Average bid at maximum karma level for different average karmas';
axes.Title.FontSize = 12;
axes.XAxis.FontSize = 10;
axes.YAxis.FontSize = 10;
axes.XLabel.FontName = 'ubuntu';
axes.XLabel.String = 'Future discount factor';
axes.XLabel.FontSize = 12;
axes.YLabel.FontName = 'ubuntu';
axes.YLabel.String = 'Average bid';
axes.YLabel.FontSize = 12;
lgd = legend(lgd_text);
lgd.FontSize = 12;
lgd.FontName = 'ubuntu';
lgd.Location = 'bestoutside';

%% Plot transition alphas
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, default_height, default_width, default_height];
plot(k_ave_vec, transition_alpha, '-x', 'LineWidth', 2);
axis tight;
axes = gca;
axes.Title.FontName = 'ubuntu';
axes.Title.String = 'Largest \alpha where maximum karma is bid at maximum level';
axes.Title.FontSize = 12;
axes.XAxis.FontSize = 10;
axes.YAxis.FontSize = 10;
axes.XLabel.FontName = 'ubuntu';
axes.XLabel.String = 'Average karma';
axes.XLabel.FontSize = 12;
axes.YLabel.FontName = 'ubuntu';
axes.YLabel.String = 'Future discount factor';
axes.YLabel.FontSize = 12;

%% Plot transition k_aves
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [default_width, default_height, default_width, default_height];
plot(alpha_vec, transition_k_ave, '-x', 'LineWidth', 2);
axis tight;
axes = gca;
axes.Title.FontName = 'ubuntu';
axes.Title.String = 'Smallest k_{ave} where maximum karma is bid at maximum level';
axes.Title.FontSize = 12;
axes.XAxis.FontSize = 10;
axes.YAxis.FontSize = 10;
axes.XLabel.FontName = 'ubuntu';
axes.XLabel.String = 'Future discount factor';
axes.XLabel.FontSize = 12;
axes.YLabel.FontName = 'ubuntu';
axes.YLabel.String = 'Average karma';
axes.YLabel.FontSize = 12;

% %% Plot slice along average karma
% alpha_slice = 0.8;
% i_alpha_slice = find(abs(alpha_vec - alpha_slice) <= eps);
% figure(fg);
% fg = fg + 1;
% fig = gcf;
% fig.Position = [0, default_height, default_width, default_height];
% plot(k_ave_vec, k_max_bid(:,i_alpha_slice), '-x', 'LineWidth', 2);
% axis tight;
% axes = gca;
% axes.Title.FontName = 'ubuntu';
% axes.Title.String = ['Average bid at maximum karma level for \alpha = ', num2str(alpha_slice, '%.2f')];
% axes.Title.FontSize = 12;
% axes.XAxis.FontSize = 10;
% axes.YAxis.FontSize = 10;
% axes.XLabel.FontName = 'ubuntu';
% axes.XLabel.String = 'Average karma';
% axes.XLabel.FontSize = 12;
% axes.YLabel.FontName = 'ubuntu';
% axes.YLabel.String = 'Average bid';
% axes.YLabel.FontSize = 12;
% 
% %% Plot slice along alpha
% k_ave_slice = 4;
% i_k_ave_slice = find(abs(k_ave_vec - k_ave_slice) <= eps);
% figure(fg);
% fg = fg + 1;
% fig = gcf;
% fig.Position = [default_width, default_height, default_width, default_height];
% plot(alpha_vec, k_max_bid(i_k_ave_slice,:), '-x', 'LineWidth', 2);
% axis tight;
% axes = gca;
% axes.Title.FontName = 'ubuntu';
% axes.Title.String = ['Average bid at maximum karma level for k_{ave} = ', num2str(k_ave_slice, '%02d')];
% axes.Title.FontSize = 12;
% axes.XAxis.FontSize = 10;
% axes.YAxis.FontSize = 10;
% axes.XLabel.FontName = 'ubuntu';
% axes.XLabel.String = 'Future discount factor';
% axes.XLabel.FontSize = 12;
% axes.YLabel.FontName = 'ubuntu';
% axes.YLabel.String = 'Average bid';
% axes.YLabel.FontSize = 12;