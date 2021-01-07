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
k_bar_vec = 10;
% alpha_comp_vec = [0.10 : 0.05 : 0.95, 0.96 : 0.01 : 0.99, 0.999, 1.00];
alpha_comp_vec = [];
z1_vec = 0 : 0.1 : 1;

n_k_bar = length(k_bar_vec);
n_alpha_comp = length(alpha_comp_vec);
n_z1 = length(z1_vec);

ne_dir = 'karma_nash_equilibrium/results/ne_U_1_10_phi1_0.50_0.50_0.50_0.50_alpha1_0.30_alpha2_0.97_pay_0/k_bar_';

sw_computed = true;
if sw_computed
    sw_dir = 'karma_nash_equilibrium/results/sw_U_1_10_phi1_0.50_0.50_0.50_0.50_pay_0/k_bar_';
end

% Flag to save plots
save_plots = true;

%% Inefficiency arrays
ne_IE = nan(n_k_bar, max([n_alpha_comp, n_z1]));
if sw_computed
    sw_IE = zeros(n_k_bar, 1);
end

for i_k_bar = 1 : n_k_bar
    k_bar = k_bar_vec(i_k_bar);
    % NE inefficiency
    ne_str = [ne_dir, num2str(k_bar, '%02d')];
    for i_alpha_comp = 1 : n_alpha_comp
        alpha = alpha_comp_vec(i_alpha_comp);
        if alpha > 0.99 && alpha < 1
            alpha_str = num2str(alpha, '%.3f');
        else
            alpha_str = num2str(alpha, '%.2f');
        end
        ne_file = [ne_str, '_alpha_', alpha_str, '.mat'];
        if exist(ne_file, 'file')
            load(ne_file, 'ne_d_up_mu_alpha_u_k', 'ne_Q_down_mu_alpha_u_k');
            ne_IE(i_k_bar,i_alpha_comp) = dot(reshape(ne_d_up_mu_alpha_u_k, [], 1), reshape(ne_Q_down_mu_alpha_u_k, [], 1));
        end
    end
    for i_z = 1 : n_z1
        ne_file = [ne_str, '_z_', num2str(z1_vec(i_z), '%.2f'), '_', num2str(1 - z1_vec(i_z), '%.2f'), '.mat'];
        if exist(ne_file, 'file')
            load(ne_file, 'ne_d_up_mu_alpha_u_k', 'ne_Q_down_mu_alpha_u_k');
            ne_IE(i_k_bar,i_z) = dot(reshape(ne_d_up_mu_alpha_u_k, [], 1), reshape(ne_Q_down_mu_alpha_u_k, [], 1));
        end
    end
    % SW inefficiency
    if sw_computed
        sw_file = [sw_dir, num2str(k_bar, '%02d'), '.mat'];
        load(sw_file, 'sw_d_up_u_k', 'sw_q_down_u_k');
        sw_IE(i_k_bar) = dot(reshape(sw_d_up_u_k, [], 1), reshape(sw_q_down_u_k, [], 1));
    end
    % Centralized urgency (best possible) efficiency (calculate once for
    % comparison)
    if i_k_bar == 1
        load(ne_file, 'param', 'ne_upsilon_up_u');
        u_IE = 0;
        for i_u = 1 : param.n_u
            u_IE = u_IE + 0.5 * ne_upsilon_up_u(i_u)^2 * param.U(i_u);
            for i_uj = i_u + 1 : param.n_u
                u_IE = u_IE + ne_upsilon_up_u(i_u) * ne_upsilon_up_u(i_uj) * param.U(i_u);
            end
        end
        rand_IE = 0.5 * dot(ne_upsilon_up_u, param.U);
    end
end

if save_plots
    if ~exist('karma_nash_equilibrium/results/ne_efficiency', 'dir')
        mkdir('karma_nash_equilibrium/results/ne_efficiency');
    end
end

%% Plot all
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
plot(k_bar_vec, repmat(-u_IE, 1, n_k_bar), 'm-', 'LineWidth', 4);
lgd_text = "centralized-urgency";
hold on;
plot(k_bar_vec, repmat(-rand_IE, 1, n_k_bar), 'r-', 'LineWidth', 4);
lgd_text = [lgd_text; "baseline-random"];
if sw_computed
    plot(k_bar_vec, -sw_IE, 'g-', 'LineWidth', 3);
    lgd_text = [lgd_text; "social-welfare"];
end
for i_alpha_comp = 1 : n_alpha_comp
    plot(k_bar_vec, -ne_IE(:,i_alpha_comp), '-x', 'LineWidth', 2);
    lgd_text = [lgd_text; strcat("$\alpha$ = ", num2str(alpha_comp_vec(i_alpha_comp), '%.2f'))];
    if m_exchange
        plot(k_bar_vec, -ne_IE_2(:,i_alpha_comp), '-o', 'LineWidth', 2);
        lgd_text = [lgd_text; strcat("$\alpha$ = ", num2str(alpha_comp_vec(i_alpha_comp), '%.2f'), "-2")];
    end
end
for i_z = 1 : n_z1
    plot(k_bar_vec, -ne_IE(:,i_z), '-x', 'LineWidth', 2);
    lgd_text = [lgd_text; strcat("$P(\alpha_{low})$ = ", num2str(z1_vec(i_z), '%.2f'))];
end
axis tight;
axes = gca;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'Efficiency';
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

if save_plots
    saveas(gcf, 'karma_nash_equilibrium/results/ne_efficiency/all.png');
end

%% Plot slices through k_bar
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
n_cols = round(sqrt(screenwidth / screenheight * (n_k_bar + 1)));
n_rows = ceil((n_k_bar + 1) / n_cols);
for i_k_bar = 1 : n_k_bar
    subplot(n_rows,n_cols,i_k_bar);
    hold on;
    if n_alpha_comp > n_z1
        semilogx(alpha_comp_vec, repmat(-u_IE, 1, n_alpha_comp), 'm-', 'LineWidth', 4);
        semilogx(alpha_comp_vec, repmat(-rand_IE, 1, n_alpha_comp), 'r-', 'LineWidth', 4);
        if sw_computed
            semilogx(alpha_comp_vec, repmat(-sw_IE(i_k_bar), 1, n_alpha_comp), 'g-', 'LineWidth', 3);
        end
        semilogx(alpha_comp_vec, -ne_IE(i_k_bar,:), '-x', 'LineWidth', 2);
    else
        plot(z1_vec, repmat(-u_IE, 1, n_z1), 'm-', 'LineWidth', 4);
        plot(z1_vec, repmat(-rand_IE, 1, n_z1), 'r-', 'LineWidth', 4);
        if sw_computed
            plot(z1_vec, repmat(-sw_IE(i_k_bar), 1, n_z1), 'g-', 'LineWidth', 3);
        end
        plot(z1_vec, -ne_IE(i_k_bar,:), '-x', 'LineWidth', 2);
    end
    axis tight;
    ylim(yl);
    axes = gca;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = ['$\bar{k}$ = ', num2str(k_bar_vec(i_k_bar), '%02d')];
    axes.Title.FontSize = 14;
    axes.XAxis.FontSize = 10;
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    if n_alpha_comp > n_z1
        axes.XLabel.String = 'Future discount factor';
    else
        axes.XLabel.String = 'Fraction of short-sighted agents';
    end
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Efficiency';
    axes.YLabel.FontSize = 12;
end
subplot(n_rows,n_cols,n_k_bar+1);
plot(nan, nan, 'm-', 'LineWidth', 4);
lgd_text = "centralized-urgency";
hold on;
plot(nan, nan, 'r-', 'LineWidth', 4);
lgd_text = [lgd_text; "baseline-random"];
if sw_computed
    plot(nan, nan, 'g-', 'LineWidth', 3);
    lgd_text = [lgd_text; "social-welfare"];
end
plot(nan, nan, '-x', 'LineWidth', 2);
lgd_text = [lgd_text; "nash-equilibrium"];
axis off;
lgd = legend(lgd_text);
lgd.FontSize = 12;
lgd.Interpreter = 'latex';
lgd.Location = 'bestoutside';
if n_alpha_comp > n_z1
    title = sgtitle('Efficiency as a function of $\alpha$ for different average karmas');
else
    title = sgtitle('Efficiency as a function of $P(\alpha_{low})$ for different average karmas');
end
title.Interpreter = 'latex';
title.FontSize = 16;

if save_plots
    saveas(gcf, 'karma_nash_equilibrium/results/ne_efficiency/k_bar_slices.png');
end

%% Plot slices through alpha/z1
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
n_subplots = max([n_alpha_comp, n_z1]) + 1;
n_cols = round(sqrt(screenwidth / screenheight * n_subplots));
n_rows = ceil(n_subplots / n_cols);
for i_subplot = 1 : max([n_alpha_comp, n_z1])
    subplot(n_rows,n_cols,i_subplot);
    plot(k_bar_vec, repmat(-u_IE, 1, n_k_bar), 'm-', 'LineWidth', 4);
    hold on;
    plot(k_bar_vec, repmat(-rand_IE, 1, n_k_bar), 'r-', 'LineWidth', 4);
    if sw_computed
        plot(k_bar_vec, -sw_IE, 'g-', 'LineWidth', 3);
    end
    plot(k_bar_vec, -ne_IE(:,i_subplot), '-x', 'LineWidth', 2);
    axis tight;
    ylim(yl);
    axes = gca;
    axes.Title.Interpreter = 'latex';
    if n_alpha_comp > n_z1
        axes.Title.String = ['$\alpha$ = ', num2str(alpha_comp_vec(i_subplot), '%0.2f')];
    else
        axes.Title.String = ['$P(\alpha_{low})$ = ', num2str(z1_vec(i_subplot), '%0.2f')];
    end
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
subplot(n_rows,n_cols,n_subplots);
plot(nan, nan, 'm-', 'LineWidth', 4);
lgd_text = "centralized-urgency";
hold on;
plot(nan, nan, 'r-', 'LineWidth', 4);
lgd_text = [lgd_text; "baseline-random"];
if sw_computed
    plot(nan, nan, 'g-', 'LineWidth', 3);
    lgd_text = [lgd_text; "social-welfare"];
end
plot(nan, nan, '-x', 'LineWidth', 2);
lgd_text = [lgd_text; "nash-equilibrium"];
axis off;
lgd = legend(lgd_text);
lgd.FontSize = 12;
lgd.Interpreter = 'latex';
lgd.Location = 'bestoutside';
if n_alpha_comp > n_z1
    title = sgtitle('Efficiency as a function of $\bar{k}$ for different future discount factors');
else
    title = sgtitle('Efficiency as a function of $\bar{k}$ for different fractions of short-sighted agents');
end
title.Interpreter = 'latex';
title.FontSize = 16;

if save_plots
    if n_alpha_comp > n_z1
        saveas(gcf, 'karma_nash_equilibrium/results/ne_efficiency/alpha_slices.png');
    else
        saveas(gcf, 'karma_nash_equilibrium/results/ne_efficiency/z1_slices.png');
    end
end