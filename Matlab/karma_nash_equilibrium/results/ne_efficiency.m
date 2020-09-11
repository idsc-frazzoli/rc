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
% k_ave_vec = 1 : 12;
k_ave_vec = 10;
% alpha_vec = [0.10 : 0.05 : 0.95, 0.96 : 0.01 : 0.99, 0.999 : 0.001 : 1.00];
% alpha_vec = [0.10 : 0.05 : 0.85, 0.86 : 0.01 : 0.90, 0.95, 0.96 : 0.01 : 0.99, 0.991 : 0.001 : 1.00];
alpha_vec = [0.10 : 0.05 : 0.95, 0.96 : 0.01 : 0.99, 0.999, 1.00];
% alpha_vec = 0.97;

num_k_ave = length(k_ave_vec);
num_alpha = length(alpha_vec);

ne_dir = 'karma_nash_equilibrium/results/ne_U_1_10_p_0.80_0.20_0.40_0.60_m_0_no_sat/k_ave_';
m_exchange = false;
if m_exchange
    ne_dir_2 = 'karma_nash_equilibrium/results/ne_U_1_2_p_0.50_m_1_no_sat/k_ave_';
end

sw_computed = true;
if sw_computed
    sw_dir = 'karma_nash_equilibrium/results/sw_U_1_10_p_0.80_0.20_0.40_0.60_m_0_no_sat/k_ave_';
    if m_exchange
        sw_dir_2 = 'karma_nash_equilibrium/results/sw_U_1_2_p_0.50_m_1_no_sat/k_ave_';
    end
end

% Flag to save plots
save_plots = false;

%% Inefficiency arrays
ne_IE = nan(num_k_ave, num_alpha);
if m_exchange
    ne_IE_2 = nan(num_k_ave, num_alpha);
end
if sw_computed
    sw_IE = zeros(num_k_ave, 1);
    if m_exchange
        sw_IE_2 = zeros(num_k_ave, 1);
    end
end

for i_k_ave = 1 : num_k_ave
    k_ave = k_ave_vec(i_k_ave);
    % NE inefficiency
    ne_str = [ne_dir, num2str(k_ave, '%02d')];
    if m_exchange
        ne_str_2 = [ne_dir_2, num2str(k_ave, '%02d')];
    end
    for i_alpha = 1 : num_alpha
        alpha = alpha_vec(i_alpha);
        if alpha > 0.99 && alpha < 1
            ne_file = [ne_str, '_alpha_', num2str(alpha, '%.3f'), '.mat'];
        else
            ne_file = [ne_str, '_alpha_', num2str(alpha, '%.2f'), '.mat'];
        end
        if exist(ne_file, 'file')
            load(ne_file, 'ne_d_up_u_k', 'ne_q_down_u_k');
            ne_IE(i_k_ave,i_alpha) = dot(reshape(ne_d_up_u_k, [], 1), reshape(ne_q_down_u_k, [], 1));
        end
        if m_exchange
            if alpha > 0.99 && alpha < 1
                ne_file_2 = [ne_str_2, '_alpha_', num2str(alpha, '%.3f'), '.mat'];
            else
                ne_file_2 = [ne_str_2, '_alpha_', num2str(alpha, '%.2f'), '.mat'];
            end
            if exist(ne_file_2, 'file')
                load(ne_file_2, 'ne_d_up_u_k', 'ne_q_down_u_k');
                ne_IE_2(i_k_ave,i_alpha) = dot(reshape(ne_d_up_u_k, [], 1), reshape(ne_q_down_u_k, [], 1));
            end
        end
    end
    % SW inefficiency
    if sw_computed
        sw_file = [sw_dir, num2str(k_ave, '%02d'), '.mat'];
        load(sw_file, 'sw_d_up_u_k', 'sw_q_down_u_k');
        sw_IE(i_k_ave) = dot(reshape(sw_d_up_u_k, [], 1), reshape(sw_q_down_u_k, [], 1));
        if m_exchange
            sw_file_2 = [sw_dir_2, num2str(k_ave, '%02d'), '.mat'];
            load(sw_file_2, 'sw_d_up_u_k', 'sw_q_down_u_k');
            sw_IE_2(i_k_ave) = dot(reshape(sw_d_up_u_k, [], 1), reshape(sw_q_down_u_k, [], 1));
        end
    end
    % Centralized urgency (best possible) efficiency (calculate once for
    % comparison)
    if i_k_ave == 1
        load(ne_file, 'ne_p_up_u', 'ne_param');
        u_IE = 0;
        for i_ui = 1 : ne_param.num_U
            u_IE = u_IE + 0.5 * ne_p_up_u(i_ui)^2 * ne_param.U(i_ui);
            for i_uj = i_ui + 1 : ne_param.num_U
                u_IE = u_IE + ne_p_up_u(i_ui) * ne_p_up_u(i_uj) * ne_param.U(i_ui);
            end
        end
        rand_IE = 0.5 * dot(ne_p_up_u, ne_param.U);
    end
end

if save_plots
    if ~exist('karma_nash_equilibrium/results/efficiency', 'dir')
        mkdir('karma_nash_equilibrium/results/efficiency');
    end
end

%% Plot all
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
plot(k_ave_vec, repmat(-u_IE, 1, num_k_ave), 'm-', 'LineWidth', 4);
lgd_text = "centralized-urgency";
hold on;
plot(k_ave_vec, repmat(-rand_IE, 1, num_k_ave), 'r-', 'LineWidth', 4);
lgd_text = [lgd_text; "baseline-random"];
if sw_computed
    plot(k_ave_vec, -sw_IE, 'g-', 'LineWidth', 3);
    lgd_text = [lgd_text; "social-welfare"];
    if m_exchange
        plot(k_ave_vec, -sw_IE_2, 'g-', 'LineWidth', 3);
        lgd_text = [lgd_text; "social-welfare-2"];
    end
end
for i_alpha = 1 : num_alpha
    plot(k_ave_vec, -ne_IE(:,i_alpha), '-x', 'LineWidth', 2);
    lgd_text = [lgd_text; strcat("$\alpha$ = ", num2str(alpha_vec(i_alpha), '%.2f'))];
    if m_exchange
        plot(k_ave_vec, -ne_IE_2(:,i_alpha), '-o', 'LineWidth', 2);
        lgd_text = [lgd_text; strcat("$\alpha$ = ", num2str(alpha_vec(i_alpha), '%.2f'), "-2")];
    end
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

if save_plots
    saveas(gcf, 'karma_nash_equilibrium/results/efficiency/all.png');
end

%% Plot slices through k_ave
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
num_cols = round(sqrt(screenwidth / screenheight * (num_k_ave + 1)));
num_rows = ceil((num_k_ave + 1) / num_cols);
for i_k_ave = 1 : num_k_ave
    subplot(num_rows,num_cols,i_k_ave);
    semilogx(alpha_vec, repmat(-u_IE, 1, num_alpha), 'm-', 'LineWidth', 4);
    hold on;
    semilogx(alpha_vec, repmat(-rand_IE, 1, num_alpha), 'r-', 'LineWidth', 4);
    if sw_computed
        semilogx(alpha_vec, repmat(-sw_IE(i_k_ave), 1, num_alpha), 'g-', 'LineWidth', 3);
        if m_exchange
            semilogx(alpha_vec, repmat(-sw_IE_2(i_k_ave), 1, num_alpha), 'm-', 'LineWidth', 3);
        end
    end
    semilogx(alpha_vec, -ne_IE(i_k_ave,:), '-x', 'LineWidth', 2);
    if m_exchange
        semilogx(alpha_vec, -ne_IE_2(i_k_ave,:), '-o', 'LineWidth', 2);
    end
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
subplot(num_rows,num_cols,num_k_ave+1);
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
title = sgtitle('Efficiency as a function of $\alpha$ for different average karmas');
title.Interpreter = 'latex';
title.FontSize = 16;

if save_plots
    saveas(gcf, 'karma_nash_equilibrium/results/efficiency/k_ave_slices.png');
end

%% Plot slices through alpha
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
num_cols = round(sqrt(screenwidth / screenheight * (num_alpha + 1)));
num_rows = ceil((num_alpha + 1) / num_cols);
for i_alpha = 1 : num_alpha
    subplot(num_rows,num_cols,i_alpha);
    plot(k_ave_vec, repmat(-u_IE, 1, num_k_ave), 'm-', 'LineWidth', 4);
    hold on;
    plot(k_ave_vec, repmat(-rand_IE, 1, num_k_ave), 'r-', 'LineWidth', 4);
    if sw_computed
        plot(k_ave_vec, -sw_IE, 'g-', 'LineWidth', 3);
        if m_exchange
            plot(k_ave_vec, -sw_IE_2, 'm-', 'LineWidth', 3);
        end
    end
    plot(k_ave_vec, -ne_IE(:,i_alpha), '-x', 'LineWidth', 2);
    if m_exchange
        plot(k_ave_vec, -ne_IE_2(:,i_alpha), '-o', 'LineWidth', 2);
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
subplot(num_rows,num_cols,num_alpha+1);
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
title = sgtitle('Efficiency as a function of $k_{avg}$ for different future discount factors');
title.Interpreter = 'latex';
title.FontSize = 16;

if save_plots
    saveas(gcf, 'karma_nash_equilibrium/results/efficiency/alpha_slices.png');
end