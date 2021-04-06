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
k_ave_vec = [1 : 20, 25 : 5 : 40];
% alpha_vec = [0.05 : 0.05 : 0.95, 0.96 : 0.01 : 1.00];
alpha_vec = 1.00;

num_k_ave = length(k_ave_vec);
num_alpha = length(alpha_vec);

file_str = 'results/N_200_T_100000_U_1_2_p_0.50_m_0_no_sat/k_ave_';

sw_computed = true;

compare = false;
if compare
    compare_file_str = 'results/N_200_T_100000_U_1_2_p_0.75_m_1_no_sat/k_ave_';
end

% Flag to save plots
save_plots = true;

%% Inefficiency arrays
ne_IE = nan(num_k_ave, num_alpha);
if sw_computed
    sw_IE = zeros(num_k_ave, 1);
end
if compare
    compare_ne_IE = nan(num_k_ave, num_alpha);
end

for i_k_ave = 1 : num_k_ave
    % NE inefficiency
    for i_alpha = 1 : num_alpha
        alpha = alpha_vec(i_alpha);
        if alpha > 0.99 && alpha < 1
            file = [file_str, num2str(k_ave_vec(i_k_ave), '%02d'), '_alpha_high.mat'];
        else
            file = [file_str, num2str(k_ave_vec(i_k_ave), '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.mat'];
        end
        if ~exist(file, 'file')
            file = [file_str, num2str(k_ave_vec(i_k_ave), '%02d'), '.mat'];
        end
        load(file, 'IE_ne', 'param');
        i_alpha_sim = find(abs(param.alpha - alpha) <= eps);
        if ~isempty(i_alpha_sim)
            ne_IE(i_k_ave,i_alpha) = IE_ne{i_alpha_sim}(end);
        end
    end
    % SW inefficiency
    if sw_computed
        load(file, 'IE_sw');
        if exist('IE_sw', 'var')
            sw_IE(i_k_ave) = IE_sw(end);
            clear IE_sw
        else
            sw_IE(i_k_ave) = nan;
        end
    end
    % Compare inefficiency
    if compare
        for i_alpha = 1 : num_alpha
            alpha = alpha_vec(i_alpha);
            if alpha > 0.99 && alpha < 1
                compare_file = [compare_file_str, num2str(k_ave_vec(i_k_ave), '%02d'), '_alpha_high.mat'];
            else
                compare_file = [compare_file_str, num2str(k_ave_vec(i_k_ave), '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.mat'];
            end
            if ~exist(compare_file, 'file')
                compare_file = [compare_file_str, num2str(k_ave_vec(i_k_ave), '%02d'), '.mat'];
            end
            load(compare_file, 'IE_ne', 'param');
            i_alpha_sim = find(abs(param.alpha - alpha) <= eps);
            if ~isempty(i_alpha_sim)
                compare_ne_IE(i_k_ave,i_alpha) = IE_ne{i_alpha_sim}(end);
            end
        end
    end
    % Centralized urgency inefficiency (get once for comparison)
    if i_k_ave == 1
        load(file, 'IE_rand', 'IE_u');
        rand_IE = IE_rand(end);
        u_IE = IE_u(end);
    end
end

if save_plots
    if ~exist('results/efficiency', 'dir')
        mkdir('results/efficiency');
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
end
for i_alpha = 1 : num_alpha
    plot(k_ave_vec, -ne_IE(:,i_alpha), '-x', 'LineWidth', 2);
    lgd_text = [lgd_text; strcat("$\alpha$ = ", num2str(alpha_vec(i_alpha), '%.2f'))];
end
if compare
    for i_alpha = 1 : num_alpha
        plot(k_ave_vec, -compare_ne_IE(:,i_alpha), '-x', 'LineWidth', 2);
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
    saveas(gcf, 'results/efficiency/all.png');
end

%% Plot slices through k_ave
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
if num_k_ave == 1
    num_cols = 1;
    num_rows = 1;
else
    num_cols = round(sqrt(screenwidth / screenheight * (num_k_ave + 1)));
    num_rows = ceil((num_k_ave + 1) / num_cols);
end
for i_k_ave = 1 : num_k_ave
    subplot(num_rows,num_cols,i_k_ave);
    semilogx(alpha_vec, repmat(-u_IE, 1, num_alpha), 'm-', 'LineWidth', 4);
    hold on;
    semilogx(alpha_vec, repmat(-rand_IE, 1, num_alpha), 'r-', 'LineWidth', 4);
    if sw_computed
        semilogx(alpha_vec, repmat(-sw_IE(i_k_ave), 1, num_alpha), 'g-', 'LineWidth', 3);
    end
    semilogx(alpha_vec, -ne_IE(i_k_ave,:), '-x', 'LineWidth', 2);
    if compare
        semilogx(alpha_vec, -compare_ne_IE(i_k_ave,:), '-o', 'LineWidth', 2);
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
if num_k_ave == 1
    lgd_text = "centralized-urgency";
    lgd_text = [lgd_text; "baseline-random"];
    if sw_computed
        lgd_text = [lgd_text; "social-welfare"];
    end
    lgd_text = [lgd_text; "nash-equilibrium"];
    if compare
        lgd_text = [lgd_text; "nash-equilibrium-2"];
    end
else
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
    if compare
        plot(nan, nan, '-o', 'LineWidth', 2);
        lgd_text = [lgd_text; "nash-equilibrium-2"];
    end
    axis off;
end
lgd = legend(lgd_text);
lgd.FontSize = 12;
lgd.Interpreter = 'latex';
lgd.Location = 'bestoutside';
title = sgtitle('Efficiency as a function of $\alpha$ for different average karmas');
title.Interpreter = 'latex';
title.FontSize = 16;

if save_plots
    saveas(gcf, 'results/efficiency/k_ave_slices.png');
end

%% Plot slices through alpha
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
if num_alpha == 1
    num_cols = 1;
    num_rows = 1;
else
    num_cols = round(sqrt(screenwidth / screenheight * (num_alpha + 1)));
    num_rows = ceil((num_alpha + 1) / num_cols);
end
for i_alpha = 1 : num_alpha
    subplot(num_rows,num_cols,i_alpha);
    plot(k_ave_vec, repmat(-u_IE, 1, num_k_ave), 'm-', 'LineWidth', 4);
    hold on;
    plot(k_ave_vec, repmat(-rand_IE, 1, num_k_ave), 'r-', 'LineWidth', 4);
    if sw_computed
        plot(k_ave_vec, -sw_IE, 'g-', 'LineWidth', 3);
    end
    plot(k_ave_vec, -ne_IE(:,i_alpha), '-x', 'LineWidth', 2);
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
if num_alpha == 1
    lgd_text = "centralized-urgency";
    lgd_text = [lgd_text; "baseline-random"];
    if sw_computed
        lgd_text = [lgd_text; "social-welfare"];
    end
    lgd_text = [lgd_text; "nash-equilibrium"];
else
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
end
lgd = legend(lgd_text);
lgd.FontSize = 12;
lgd.Interpreter = 'latex';
lgd.Location = 'bestoutside';
title = sgtitle('Efficiency as a function of $k_{avg}$ for different future discount factors');
title.Interpreter = 'latex';
title.FontSize = 16;

if save_plots
    saveas(gcf, 'results/efficiency/alpha_slices.png');
end