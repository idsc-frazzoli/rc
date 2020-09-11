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
cmap_parula = parula(round(1.2 * param.num_alpha));
cmap_parula(param.num_alpha+1:end,:) = [];
cmap_parula = flip(cmap_parula);
cmap_autumn = autumn(round(1.2 * param.num_lim_mem_steps));

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
if control.lim_mem_policies
    set(gca, 'ColorOrder', cmap_autumn);
    set(gca, 'ColorOrderIndex', 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        pl = plot(-IE_lim_mem_a{i_lim_mem}(end), -UF_lim_mem_a{i_lim_mem}(end),...
            'LineStyle', 'none',...
            'LineWidth', 1.5,...
            'Marker', '+',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        if control.compute_entropy
            lgd_text = [lgd_text, strcat("cent-cost-", int2str(param.lim_mem_steps(i_lim_mem)), ", ", num2str(ent_lim_mem_a(i_lim_mem), '%.2f'), "-bit")];
        else
            lgd_text = [lgd_text, strcat("cent-cost-", int2str(param.lim_mem_steps(i_lim_mem)))];
        end
    end
    set(gca, 'ColorOrderIndex', 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        pl = plot(-IE_lim_mem_u_a{i_lim_mem}(end), -UF_lim_mem_u_a{i_lim_mem}(end),...
            'LineStyle', 'none',...
            'LineWidth', 1.5,...
            'Marker', '*',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        if control.compute_entropy
            lgd_text = [lgd_text, strcat("cent-urg-th-cost-", int2str(param.lim_mem_steps(i_lim_mem)), ", ", num2str(ent_lim_mem_u_a(i_lim_mem), '%.2f'), "-bit")];
        else
            lgd_text = [lgd_text, strcat("cent-urg-th-cost-", int2str(param.lim_mem_steps(i_lim_mem)))];
        end
    end
end
if control.karma_heuristic_policies
    set(gca, 'ColorOrder', cmap);
    set(gca, 'ColorOrderIndex', 1);
    pl = plot(-IE_bid_1(end), -UF_bid_1(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    if control.compute_entropy
        lgd_text = [lgd_text, strcat("bid-1, ", num2str(ent_bid_1, '%.2f'), "-bit")];
    else
        lgd_text = [lgd_text, "bid-1"];
    end
    pl = plot(-IE_bid_u(end), -UF_bid_u(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    if control.compute_entropy
        lgd_text = [lgd_text, strcat("bid-urg, ", num2str(ent_bid_u, '%.2f'), "-bit")];
    else
        lgd_text = [lgd_text, "bid-urg"];
    end
    pl = plot(-IE_bid_all(end), -UF_bid_all(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    if control.compute_entropy
        lgd_text = [lgd_text, strcat("bid-all, ", num2str(ent_bid_all, '%.2f'), "-bit")];
    else
        lgd_text = [lgd_text, "bid-all"];
    end
    pl = plot(-IE_bid_all_u(end), -UF_bid_all_u(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    if control.compute_entropy
        lgd_text = [lgd_text, strcat("bid-all-if-urg, ", num2str(ent_bid_all_u, '%.2f'), "-bit")];
    else
        lgd_text = [lgd_text, "bid-all-if-urg"];
    end
    pl = plot(-IE_bid_rand(end), -UF_bid_rand(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    if control.compute_entropy
        lgd_text = [lgd_text, strcat("bid-rand, ", num2str(ent_bid_rand, '%.2f'), "-bit")];
    else
        lgd_text = [lgd_text, "bid-random"];
    end
    pl = plot(-IE_bid_rand_u(end), -UF_bid_rand_u(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    if control.compute_entropy
        lgd_text = [lgd_text, strcat("bid-random-if-urg, ", num2str(ent_bid_rand_u, '%.2f'), "-bit")];
    else
        lgd_text = [lgd_text, "bid-random-if-urg"];
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
    for i_alpha = 1 : param.num_alpha
        pl = plot(-IE_ne{i_alpha}(end), -UF_ne{i_alpha}(end),...
            'LineStyle', 'none',...
            'Marker', 'o',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        if control.compute_entropy
            lgd_text = [lgd_text, strcat("$\alpha$ = ", num2str(param.alpha(i_alpha), '%.3f'), ", ", num2str(ent_ne(i_alpha), '%.2f'), "-bit")];
        else
            lgd_text = [lgd_text, strcat("$\alpha$ = ", num2str(param.alpha(i_alpha), '%.3f'))];
        end
    end
end
axes = gca;
func.axis_semi_tight(axes, 1.2);
axes.Title.Interpreter = 'latex';
if control.karma_heuristic_policies || control.karma_ne_policies || control.karma_sw_policy
    axes.Title.String = ['$k_{avg}$ = ', num2str(param.k_ave, '%02d'), ' performance comparison'];
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
% axes.YLabel.String = 'Fairness (negative variance of relative payoff)';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.Interpreter = 'latex';
lgd.FontSize = 12;
lgd.Location = 'bestoutside';

if param.plot_a
    %% Accumulated cost plot - Gives indication on how fast variance grows
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [mod(vert_fg,2)*default_width, 0, default_width, screenheight];
    vert_fg = vert_fg + 1;
    subplot(2,2,1);
    hold off;
    plot(a_rand);
    hold on;
    plot(IE_rand, 'Linewidth', 3);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'random';
    axes.Title.FontSize = 14;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,2);
    hold off;
    plot(a_u);
    hold on;
    plot(IE_u, 'Linewidth', 3);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'cent-urg';
    axes.Title.FontSize = 14;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,3);
    hold off;
    plot(a_a);
    hold on;
    plot(IE_a, 'Linewidth', 3);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'cent-cost';
    axes.Title.FontSize = 14;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,4);
    hold off;
    plot(a_u_a);
    hold on;
    plot(IE_u_a, 'Linewidth', 3);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'cent-urg-th-cost';
    axes.Title.FontSize = 14;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Accumulated cost';
    axes.YLabel.FontSize = 12;
    
    title = sgtitle('Accumulated costs for centralized policies');
    title.Interpreter = 'latex';
    title.FontSize = 16;

    %% Accumulated cost plot for limited memory policies
    if control.lim_mem_policies
        figure(fg);
        fg = fg + 1;
        fig = gcf;
        fig.Position = [0, 0, screenwidth, screenheight];
        num_cols = round(sqrt(screenwidth / screenheight * param.num_lim_mem_steps));
        num_rows = ceil(param.num_lim_mem_steps / num_cols);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            subplot(num_rows,num_cols,i_lim_mem);
            hold off;
            plot(a_lim_mem_a{i_lim_mem});
            hold on;
            plot(IE_lim_mem_a{i_lim_mem}, 'Linewidth', 3);
            axes = gca;
            axis tight;
            axes.Title.Interpreter = 'latex';
            axes.Title.String = ['cent-cost-', int2str(param.lim_mem_steps(i_lim_mem))];
            axes.Title.FontSize = 14;
            axes.XAxis.TickLabelInterpreter = 'latex';
            axes.XAxis.FontSize = 10;
            axes.YAxis.TickLabelInterpreter = 'latex';
            axes.YAxis.FontSize = 10;
            axes.XLabel.Interpreter = 'latex';
            axes.XLabel.String = 'Time';
            axes.XLabel.FontSize = 12;
            axes.YLabel.Interpreter = 'latex';
            axes.YLabel.String = 'Accumulated cost';
            axes.YLabel.FontSize = 12;
        end
        title = sgtitle('Accumulated costs for centralized cost with limited memory policies');
        title.Interpreter = 'latex';
        title.FontSize = 16;
        
        figure(fg);
        fg = fg + 1;
        fig = gcf;
        fig.Position = [0, 0, screenwidth, screenheight];
        num_cols = round(sqrt(screenwidth / screenheight * param.num_lim_mem_steps));
        num_rows = ceil(param.num_lim_mem_steps / num_cols);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            subplot(num_rows,num_cols,i_lim_mem);
            hold off;
            plot(a_lim_mem_u_a{i_lim_mem});
            hold on;
            plot(IE_lim_mem_u_a{i_lim_mem}, 'Linewidth', 3);
            axes = gca;
            axis tight;
            axes.Title.Interpreter = 'latex';
            axes.Title.String = ['cent-urg-th-cost-', int2str(param.lim_mem_steps(i_lim_mem))];
            axes.Title.FontSize = 14;
            axes.XAxis.TickLabelInterpreter = 'latex';
            axes.XAxis.FontSize = 10;
            axes.YAxis.TickLabelInterpreter = 'latex';
            axes.YAxis.FontSize = 10;
            axes.XLabel.Interpreter = 'latex';
            axes.XLabel.String = 'Time';
            axes.XLabel.FontSize = 12;
            axes.YLabel.Interpreter = 'latex';
            axes.YLabel.String = 'Accumulated cost';
            axes.YLabel.FontSize = 12;
        end
        title = sgtitle('Accumulated costs for centralized urgency then cost with limited memory policies');
        title.Interpreter = 'latex';
        title.FontSize = 16;
    end

    %% Accumulated cost plot for heuristic karma policies
    if control.karma_heuristic_policies
        figure(fg);
        fg = fg + 1;
        fig = gcf;
        fig.Position = [mod(vert_fg,2)*default_width, 0, default_width, screenheight];
        vert_fg = vert_fg + 1;
        subplot(3,2,1);
        hold off;
        plot(a_bid_1);
        hold on;
        plot(IE_bid_1, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-1';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,2);
        hold off;
        plot(a_bid_u);
        hold on;
        plot(IE_bid_u, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-urg';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,3);
        hold off;
        plot(a_bid_all);
        hold on;
        plot(IE_bid_all, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-all';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,4);
        hold off;
        plot(a_bid_all_u);
        hold on;
        plot(IE_bid_all_u, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-all-if-urg';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,5);
        hold off;
        plot(a_bid_rand);
        hold on;
        plot(IE_bid_rand, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-random';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,6);
        hold off;
        plot(a_bid_rand_u);
        hold on;
        plot(IE_bid_rand_u, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-random-if-urg';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Accumulated cost';
        axes.YLabel.FontSize = 12;
        
        title = sgtitle(['$k_{avg}$ = ', num2str(param.k_ave, '%02d'), ' accumulated costs for heuristic karma policies']);
        title.Interpreter = 'latex';
        title.FontSize = 16;
    end

    %% Accumulated cost plot for karma NE and/or SW policies
    if control.karma_ne_policies || control.karma_sw_policy
        figure(fg);
        fg = fg + 1;
        fig = gcf;
        fig.Position = [0, 0, screenwidth, screenheight];
        num_subplots = 0;
        if control.karma_ne_policies
            num_subplots = num_subplots + param.num_alpha;
        end
        if control.karma_sw_policy
            num_subplots = num_subplots + 1;
        end
        num_cols = round(sqrt(screenwidth / screenheight * num_subplots));
        num_rows = ceil(num_subplots / num_cols);
        if control.karma_ne_policies
            for i_subplot = 1 : param.num_alpha
                subplot(num_rows,num_cols,i_subplot);
                hold off;
                plot(a_ne{i_subplot});
                hold on;
                plot(IE_ne{i_subplot}, 'Linewidth', 3);
                axes = gca;
                axis tight;
                axes.Title.Interpreter = 'latex';
                axes.Title.String = ['$\alpha$ = ', num2str(param.alpha(i_subplot), '%.3f')];
                axes.Title.FontSize = 14;
                axes.XAxis.TickLabelInterpreter = 'latex';
                axes.XAxis.FontSize = 10;
                axes.YAxis.TickLabelInterpreter = 'latex';
                axes.YAxis.FontSize = 10;
                axes.XLabel.Interpreter = 'latex';
                axes.XLabel.String = 'Time';
                axes.XLabel.FontSize = 12;
                axes.YLabel.Interpreter = 'latex';
                axes.YLabel.String = 'Accumulated cost';
                axes.YLabel.FontSize = 12;
            end
        else
            i_subplot = 1;
        end
        if control.karma_sw_policy
            subplot(num_rows,num_cols,i_subplot);
            hold off;
            plot(a_sw);
            hold on;
            plot(IE_sw, 'Linewidth', 3);
            axes = gca;
            axis tight;
            axes.Title.Interpreter = 'latex';
            axes.Title.String = 'social-welfare';
            axes.Title.FontSize = 14;
            axes.XAxis.TickLabelInterpreter = 'latex';
            axes.XAxis.FontSize = 10;
            axes.YAxis.TickLabelInterpreter = 'latex';
            axes.YAxis.FontSize = 10;
            axes.XLabel.Interpreter = 'latex';
            axes.XLabel.String = 'Time';
            axes.XLabel.FontSize = 12;
            axes.YLabel.Interpreter = 'latex';
            axes.YLabel.String = 'Accumulated cost';
            axes.YLabel.FontSize = 12;
        end
        
        title = sgtitle(['$k_{avg}$ = ', num2str(param.k_ave, '%02d'), ' accumulated costs for karma policies']);
        title.Interpreter = 'latex';
        title.FontSize = 16;
    end
end

if param.plot_F
    %% Fairness vs. time
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [0, 0, screenwidth, screenheight];
    hold off;
    plot(-UF_rand, 'LineWidth', 2);
    hold on;
    plot(-UF_u, 'LineWidth', 2);
    plot(-UF_a, 'LineWidth', 2);
    plot(-UF_u_a, 'LineWidth', 2);
    if control.lim_mem_policies
        for i_lim_mem = 1 : param.num_lim_mem_steps
            plot(-UF_lim_mem_a{i_lim_mem});
        end
        for i_lim_mem = 1 : param.num_lim_mem_steps
            plot(-UF_lim_mem_u_a{i_lim_mem}, ':');
        end
    end
    if control.karma_heuristic_policies
        plot(-UF_bid_1, '-.');
        plot(-UF_bid_u, '-.');
        plot(-UF_bid_all, '-.');
        plot(-UF_bid_all_u, '-.');
        plot(-UF_bid_rand, '-.');
        plot(-UF_bid_rand_u, '-.');
    end
    if control.karma_sw_policy
        plot(-UF_sw, '--', 'LineWidth', 2);
    end
    if control.karma_ne_policies
        for i_alpha = 1 : param.num_alpha
            plot(-UF_ne{i_alpha}, '--');
        end
    end
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    if control.karma_heuristic_policies || control.karma_ne_policies || control.karma_sw_policy
        axes.Title.String = ['$k_{avg}$ = ', num2str(param.k_ave, '%02d'), ' fairness vs. time'];
    else
        axes.Title.String = 'Fairness vs. time';
    end
    axes.Title.FontSize = 16;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time';
    axes.XLabel.FontSize = 14;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Fairness';
    axes.YLabel.FontSize = 14;
    lgd = legend(lgd_text);
    lgd.Interpreter = 'latex';
    lgd.FontSize = 12;
    lgd.Location = 'bestoutside';
end

if control.compute_a_acorr && param.plot_a_std
    %% Standardized accumulated cost plot
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [mod(vert_fg,2)*default_width, 0, default_width, screenheight];
    vert_fg = vert_fg + 1;
    subplot(2,2,1);
    plot(a_rand_std);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'random';
    axes.Title.FontSize = 14;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Standardized accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,2);
    plot(a_u_std);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'cent-urg';
    axes.Title.FontSize = 14;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Standardized accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,3);
    plot(a_a_std);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'cent-cost';
    axes.Title.FontSize = 14;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Standardized accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,4);
    plot(a_u_a_std);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'cent-urg-th-cost';
    axes.Title.FontSize = 14;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time';
    axes.XLabel.FontSize = 12;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Standardized accumulated cost';
    axes.YLabel.FontSize = 12;
    
    title = sgtitle('Standardized accumulated costs for centralized policies');
    title.Interpreter = 'latex';
    title.FontSize = 16;

    %% Standardized accumulated cost plot for limited memory policies
    if control.lim_mem_policies
        figure(fg);
        fg = fg + 1;
        fig = gcf;
        fig.Position = [0, 0, screenwidth, screenheight];
        num_cols = round(sqrt(screenwidth / screenheight * param.num_lim_mem_steps));
        num_rows = ceil(param.num_lim_mem_steps / num_cols);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            subplot(num_rows,num_cols,i_lim_mem);
            plot(a_lim_mem_a_std{i_lim_mem});
            axes = gca;
            axis tight;
            axes.Title.Interpreter = 'latex';
            axes.Title.String = ['cent-cost-', int2str(param.lim_mem_steps(i_lim_mem))];
            axes.Title.FontSize = 14;
            axes.XAxis.TickLabelInterpreter = 'latex';
            axes.XAxis.FontSize = 10;
            axes.YAxis.TickLabelInterpreter = 'latex';
            axes.YAxis.FontSize = 10;
            axes.XLabel.Interpreter = 'latex';
            axes.XLabel.String = 'Time';
            axes.XLabel.FontSize = 12;
            axes.YLabel.Interpreter = 'latex';
            axes.YLabel.String = 'Standardized accumulated cost';
            axes.YLabel.FontSize = 12;
        end
        title = sgtitle('Standardized accumulated costs for centralized cost with limited memory policies');
        title.Interpreter = 'latex';
        title.FontSize = 16;
        
        figure(fg);
        fg = fg + 1;
        fig = gcf;
        fig.Position = [0, 0, screenwidth, screenheight];
        num_cols = round(sqrt(screenwidth / screenheight * param.num_lim_mem_steps));
        num_rows = ceil(param.num_lim_mem_steps / num_cols);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            subplot(num_rows,num_cols,i_lim_mem);
            plot(a_lim_mem_u_a_std{i_lim_mem});
            axes = gca;
            axis tight;
            axes.Title.Interpreter = 'latex';
            axes.Title.String = ['cent-urg-th-cost-', int2str(param.lim_mem_steps(i_lim_mem))];
            axes.Title.FontSize = 14;
            axes.XAxis.TickLabelInterpreter = 'latex';
            axes.XAxis.FontSize = 10;
            axes.YAxis.TickLabelInterpreter = 'latex';
            axes.YAxis.FontSize = 10;
            axes.XLabel.Interpreter = 'latex';
            axes.XLabel.String = 'Time';
            axes.XLabel.FontSize = 12;
            axes.YLabel.Interpreter = 'latex';
            axes.YLabel.String = 'Standardized accumulated cost';
            axes.YLabel.FontSize = 12;
        end
        title = sgtitle('Standardized accumulated costs for centralized urgency then cost with limited memory policies');
        title.Interpreter = 'latex';
        title.FontSize = 16;
    end

    %% Standardized accumulated cost plot for heuristic karma policies
    if control.karma_heuristic_policies
        figure(fg);
        fg = fg + 1;
        fig = gcf;
        fig.Position = [mod(vert_fg,2)*default_width, 0, default_width, screenheight];
        vert_fg = vert_fg + 1;
        subplot(3,2,1);
        plot(a_bid_1_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-1';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Standardized accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,2);
        plot(a_bid_u_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-urg';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Standardized accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,3);
        plot(a_bid_all_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-all';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Standardized accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,4);
        plot(a_bid_all_u_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-all-if-urg';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Standardized accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,5);
        plot(a_bid_rand_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-random';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Standardized accumulated cost';
        axes.YLabel.FontSize = 12;
        subplot(3,2,6);
        plot(a_bid_rand_u_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-random-if-urg';
        axes.Title.FontSize = 14;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Standardized accumulated cost';
        axes.YLabel.FontSize = 12;
        
        title = sgtitle(['$k_{avg}$ = ', num2str(param.k_ave, '%02d'), ' standardized accumulated costs for heuristic karma policies']);
        title.Interpreter = 'latex';
        title.FontSize = 16;
    end
    
    %% Standardized accumulated cost plot for karma NE and/or SW policies
    if control.karma_ne_policies || control.karma_sw_policy
        figure(fg);
        fg = fg + 1;
        fig = gcf;
        fig.Position = [0, 0, screenwidth, screenheight];
        num_subplots = 0;
        if control.karma_ne_policies
            num_subplots = num_subplots + param.num_alpha;
        end
        if control.karma_sw_policy
            num_subplots = num_subplots + 1;
        end
        num_cols = round(sqrt(screenwidth / screenheight * num_subplots));
        num_rows = ceil(num_subplots / num_cols);
        if control.karma_ne_policies
            for i_subplot = 1 : param.num_alpha
                subplot(num_rows,num_cols,i_subplot);
                plot(a_ne_std{i_subplot});
                axes = gca;
                axis tight;
                axes.Title.Interpreter = 'latex';
                axes.Title.String = ['$\alpha$ = ', num2str(param.alpha(i_subplot), '%.3f')];
                axes.Title.FontSize = 14;
                axes.XAxis.TickLabelInterpreter = 'latex';
                axes.XAxis.FontSize = 10;
                axes.YAxis.TickLabelInterpreter = 'latex';
                axes.YAxis.FontSize = 10;
                axes.XLabel.Interpreter = 'latex';
                axes.XLabel.String = 'Time';
                axes.XLabel.FontSize = 12;
                axes.YLabel.Interpreter = 'latex';
                axes.YLabel.String = 'Accumulated cost';
                axes.YLabel.FontSize = 12;
            end
        else
            i_subplot = 1;
        end
        if control.karma_sw_policy
            subplot(num_rows,num_cols,i_subplot);
            plot(a_sw_std);
            axes = gca;
            axis tight;
            axes.Title.Interpreter = 'latex';
            axes.Title.String = 'social-welfare';
            axes.Title.FontSize = 14;
            axes.XAxis.TickLabelInterpreter = 'latex';
            axes.XAxis.FontSize = 10;
            axes.YAxis.TickLabelInterpreter = 'latex';
            axes.YAxis.FontSize = 10;
            axes.XLabel.Interpreter = 'latex';
            axes.XLabel.String = 'Time';
            axes.XLabel.FontSize = 12;
            axes.YLabel.Interpreter = 'latex';
            axes.YLabel.String = 'Accumulated cost';
            axes.YLabel.FontSize = 12;
        end
        
        title = sgtitle(['$k_{avg}$ = ', num2str(param.k_ave, '%02d'), ' standardized accumulated costs for karma policies']);
        title.Interpreter = 'latex';
        title.FontSize = 16;
    end
end

if control.compute_a_acorr && param.plot_a_acorr
    %% Autocorrelation of accumulated costs
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [0, 0, screenwidth, screenheight];
    hold off;
    plot(acorr_tau, a_rand_acorr, 'LineWidth', 2);
    hold on;
    plot(acorr_tau, a_u_acorr, 'LineWidth', 2);
    plot(acorr_tau, a_a_acorr, 'LineWidth', 2);
    plot(acorr_tau, a_u_a_acorr, 'LineWidth', 2);
    if control.lim_mem_policies
        for i_lim_mem = 1 : param.num_lim_mem_steps
            plot(acorr_tau, a_lim_mem_a_acorr{i_lim_mem});
        end
        for i_lim_mem = 1 : param.num_lim_mem_steps
            plot(acorr_tau, a_lim_mem_u_a_acorr{i_lim_mem}, ':');
        end
    end
    if control.karma_heuristic_policies
        plot(acorr_tau, a_bid_1_acorr, '-.');
        plot(acorr_tau, a_bid_u_acorr, '-.');
        plot(acorr_tau, a_bid_all_acorr, '-.');
        plot(acorr_tau, a_bid_all_u_acorr, '-.');
        plot(acorr_tau, a_bid_rand_acorr, '-.');
        plot(acorr_tau, a_bid_rand_u_acorr, '-.');
    end
    if control.karma_sw_policy
        plot(acorr_tau, a_sw_acorr, '--', 'LineWidth', 2);
    end
    if control.karma_ne_policies
        for i_alpha = 1 : param.num_alpha
            plot(acorr_tau, a_ne_acorr{i_alpha}, '--');
        end
    end
    axis tight;
    axes = gca;
    yl = ylim(axes);
    stem(0, a_rand_acorr(acorr_tau == 0));
    ylim(axes, yl);
    axes.Title.Interpreter = 'latex';
    if control.karma_heuristic_policies || control.karma_ne_policies || control.karma_sw_policy
        axes.Title.String = ['$k_{avg}$ = ', num2str(param.k_ave, '%02d'), ' autocorrelation of accumulated costs'];
    else
        axes.Title.String = 'Autocorrelation of accumulated costs';
    end
    axes.Title.FontSize = 16;
    axes.XAxis.TickLabelInterpreter = 'latex';
    axes.XAxis.FontSize = 10;
    axes.YAxis.TickLabelInterpreter = 'latex';
    axes.YAxis.FontSize = 10;
    axes.XLabel.Interpreter = 'latex';
    axes.XLabel.String = 'Time shift';
    axes.XLabel.FontSize = 14;
    axes.YLabel.Interpreter = 'latex';
    axes.YLabel.String = 'Autocorrelation';
    axes.YLabel.FontSize = 14;
    lgd = legend(lgd_text);
    lgd.Interpreter = 'latex';
    lgd.FontSize = 12;
    lgd.Location = 'bestoutside';
end