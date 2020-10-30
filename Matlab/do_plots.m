%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
default_width = screenwidth / 2;
default_height = screenheight / 2;
fg = 1;
vert_fg = 0;

%% Set the Color order
cmap = unique(lines, 'rows');
cmap = [cmap; unique(prism, 'rows')];
cmap(end,:) = [];
cmap_end = colorcube;
cmap_end(39:end,:) = [];
rng(1000);
cmap_end = cmap_end(randperm(size(cmap_end,1)),:);
cmap = [cmap; cmap_end];
cmap_parula = parula(round(1.2 * param.n_alpha_comp));
cmap_parula(param.n_alpha_comp+1:end,:) = [];
cmap_parula = flip(cmap_parula);
cmap_autumn = autumn(round(1.2 * param.n_fairness_horizon));

%% Scatter plot - Efficiency vs fairness
performance_comparison_fg = fg;
figure(performance_comparison_fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
set(fig, 'defaultAxesColorOrder', cmap);
hold off;
pl = plot(-IE_rand(end), -UF_rand(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = "random";
hold on;
pl = plot(-IE_u(end), -UF_u(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = [lgd_text, "cent-urg"];
pl = plot(-IE_a(end), -UF_a(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = [lgd_text, "cent-cost"];
pl = plot(-IE_u_a(end), -UF_u_a(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = [lgd_text, "cent-urg-th-cost"];
if control.fairness_horizon_policies
    set(gca, 'ColorOrder', cmap_autumn);
    set(gca, 'ColorOrderIndex', 1);
    for i_fair_hor = 1 : param.n_fairness_horizon
        pl = plot(-IE_fair_hor_a{i_fair_hor}(end), -UF_fair_hor_a{i_fair_hor}(end),...
            'LineStyle', 'none',...
            'LineWidth', 1.5,...
            'Marker', '+',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        if control.compute_entropy
            lgd_text = [lgd_text, strcat("cent-cost-", int2str(param.fairness_horizon(i_fair_hor)), ", ", num2str(ent_fair_hor_a(i_fair_hor), '%.2f'), "-bit")];
        else
            lgd_text = [lgd_text, strcat("cent-cost-", int2str(param.fairness_horizon(i_fair_hor)))];
        end
    end
    set(gca, 'ColorOrderIndex', 1);
    for i_fair_hor = 1 : param.n_fairness_horizon
        pl = plot(-IE_fair_hor_u_a{i_fair_hor}(end), -UF_fair_hor_u_a{i_fair_hor}(end),...
            'LineStyle', 'none',...
            'LineWidth', 1.5,...
            'Marker', '*',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        if control.compute_entropy
            lgd_text = [lgd_text, strcat("cent-urg-th-cost-", int2str(param.fairness_horizon(i_fair_hor)), ", ", num2str(ent_fair_hor_u_a(i_fair_hor), '%.2f'), "-bit")];
        else
            lgd_text = [lgd_text, strcat("cent-urg-th-cost-", int2str(param.fairness_horizon(i_fair_hor)))];
        end
    end
end
if control.karma_sw_policy
    set(gca, 'ColorOrder', cmap);
    set(gca, 'ColorOrderIndex', 1);
    pl = plot(-IE_sw(end), -UF_sw(end),...
        'LineStyle', 'none',...
        'Marker', 's',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    if control.compute_entropy
        lgd_text = [lgd_text, strcat("social-welfare, ", num2str(ent_sw, '%.2f'), "-bit")];
    else
        lgd_text = [lgd_text, "social-welfare"];
    end
end
if control.karma_ne_policies
    set(gca, 'ColorOrder', cmap_parula);
    set(gca, 'ColorOrderIndex', 1);
    for i_alpha_comp = 1 : param.n_alpha_comp
        pl = plot(-IE_ne{i_alpha_comp}(end), -UF_ne{i_alpha_comp}(end),...
            'LineStyle', 'none',...
            'Marker', 'o',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        if control.compute_entropy
            lgd_text = [lgd_text, strcat("$\alpha$ = ", num2str(param.Alpha(i_alpha_comp), '%.3f'), ", ", num2str(ent_ne(i_alpha_comp), '%.2f'), "-bit")];
        else
            lgd_text = [lgd_text, strcat("$\alpha$ = ", num2str(param.Alpha(i_alpha_comp), '%.3f'))];
        end
    end
end
axes = gca;
axis_semi_tight(axes, 1.2);
axes.Title.Interpreter = 'latex';
if control.karma_ne_policies || control.karma_sw_policy
    axes.Title.String = ['$\bar{k}$ = ', num2str(param.k_bar, '%02d'), ' performance comparison'];
else
    axes.Title.String = 'Performance comparison';
end
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Efficiency (mean of payoff)';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Fairness (negative variance of payoff)';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.Interpreter = 'latex';
lgd.FontSize = 12;
lgd.Location = 'bestoutside';