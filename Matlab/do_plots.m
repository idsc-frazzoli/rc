%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
default_width = screenwidth / 2;
default_height = screenheight / 2;
fg = 1;
vert_fg = 0;

%% Scatter plot - Efficiency vs fairness
performance_comparison_fg = fg;
figure(performance_comparison_fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
hold off;
pl = plot(-W1_rand(end), -W2_rand(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = "baseline-random";
hold on;
pl = plot(-W1_1(end), -W2_1(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = [lgd_text, "centralized-urgency"];
pl = plot(-W1_2(end), -W2_2(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = [lgd_text, "centralized-cost"];
pl = plot(-W1_1_2(end), -W2_1_2(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = [lgd_text, "centralized-urgency-then-cost"];
if control.lim_mem_policies
    for i_lim_mem = 1 : param.num_lim_mem_steps
        pl = plot(-W1_lim_mem_u{i_lim_mem}(end), -W2_lim_mem_u{i_lim_mem}(end),...
            'LineStyle', 'none',...
            'Marker', '*',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        lgd_text = [lgd_text, strcat("centralized-urgency-then-cost-mem-", int2str(param.lim_mem_steps(i_lim_mem)))];
    end
end
if control.karma_heuristic_policies
    pl = plot(-W1_bid_1(end), -W2_bid_1(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "bid-1-always"];
    pl = plot(-W1_bid_1_u(end), -W2_bid_1_u(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "bid-1-if-urgent"];
    pl = plot(-W1_bid_all(end), -W2_bid_all(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "bid-all-always"];
    pl = plot(-W1_bid_all_u(end), -W2_bid_all_u(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "bid-all-if-urgent"];
end
if control.karma_ne_policies
    for i_alpha = 1 : param.num_alpha
        pl = plot(-W1_ne{i_alpha}(end), -W2_ne{i_alpha}(end),...
            'LineStyle', 'none',...
            'Marker', 'o',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        lgd_text = [lgd_text, strcat("$\alpha$ = ", num2str(param.alpha(i_alpha), '%.2f'))];
    end
end
if control.karma_sw_policy
    pl = plot(-W1_sw(end), -W2_sw(end),...
        'LineStyle', 'none',...
        'Marker', 's',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "social-welfare"];
end
axes = gca;
func.axis_semi_tight(axes, 1.2);
axes.Title.Interpreter = 'latex';
axes.Title.String = ['$k_{max}$ = ', num2str(param.k_max, '%02d'), ' $k_{ave}$ = ', num2str(param.k_ave, '%02d'), ' performance comparison'];
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Efficiency (mean of reward)';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Fairness (negative variance of reward)';
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
    plot(W1_rand, 'Linewidth', 3);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'baseline-random';
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
    plot(a_1);
    hold on;
    plot(W1_1, 'Linewidth', 3);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'centralized-urgency';
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
    plot(a_2);
    hold on;
    plot(W1_2, 'Linewidth', 3);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'centralized-cost';
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
    plot(a_1_2);
    hold on;
    plot(W1_1_2, 'Linewidth', 3);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'centralized-urgency-then-cost';
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
            plot(a_lim_mem_u{i_lim_mem});
            hold on;
            plot(W1_lim_mem_u{i_lim_mem}, 'Linewidth', 3);
            axes = gca;
            axis tight;
            axes.Title.Interpreter = 'latex';
            axes.Title.String = ['centralized-urgency-then-cost-mem-', int2str(param.lim_mem_steps(i_lim_mem))];
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
        
        title = sgtitle('Accumulated costs for limited memory policies');
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
        subplot(2,2,1);
        hold off;
        plot(a_bid_1);
        hold on;
        plot(W1_bid_1, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-1-always';
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
        plot(a_bid_1_u);
        hold on;
        plot(W1_bid_1_u, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-1-if-urgent';
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
        plot(a_bid_all);
        hold on;
        plot(W1_bid_all, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-all-always';
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
        plot(a_bid_all_u);
        hold on;
        plot(W1_bid_all_u, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-all-if-urgent';
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
        
        title = sgtitle(['$k_{max}$ = ', num2str(param.k_max, '%02d'), ' $k_{ave}$ = ', num2str(param.k_ave, '%02d'), ' accumulated costs for heuristic karma policies']);
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
                plot(W1_ne{i_subplot}, 'Linewidth', 3);
                axes = gca;
                axis tight;
                axes.Title.Interpreter = 'latex';
                axes.Title.String = ['$\alpha$ = ', num2str(param.alpha(i_subplot), '%.2f')];
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
            plot(W1_sw, 'Linewidth', 3);
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
        
        title = sgtitle(['$k_{max}$ = ', num2str(param.k_max, '%02d'), ' $k_{ave}$ = ', num2str(param.k_ave, '%02d'), ' accumulated costs for karma policies']);
        title.Interpreter = 'latex';
        title.FontSize = 16;
    end
end

if param.plot_W2
    %% Fairness vs. time
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [0, 0, screenwidth, screenheight];
    hold off;
    plot(-W2_rand, 'LineWidth', 2);
    hold on;
    plot(-W2_1, 'LineWidth', 2);
    plot(-W2_2, 'LineWidth', 2);
    plot(-W2_1_2, 'LineWidth', 2);
    if control.lim_mem_policies
        for i_lim_mem = 1 : param.num_lim_mem_steps
            plot(-W2_lim_mem_u{i_lim_mem}, ':');
        end
    end
    if control.karma_heuristic_policies
        plot(-W2_bid_1, '-.');
        plot(-W2_bid_1_u, '-.');
        plot(-W2_bid_all, '-.');
        plot(-W2_bid_all_u, '-.');
    end
    if control.karma_ne_policies
        for i_alpha = 1 : param.num_alpha
            plot(-W2_ne{i_alpha}, '--');
        end
    end
    if control.karma_sw_policy
        plot(-W2_sw, '--', 'LineWidth', 2);
    end
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = ['$k_{max}$ = ', num2str(param.k_max, '%02d'), ' $k_{ave}$ = ', num2str(param.k_ave, '%02d'), ' fairness vs. time'];
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
    axes.Title.String = 'baseline-random';
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
    plot(a_1_std);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'centralized-urgency';
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
    plot(a_2_std);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'centralized-cost';
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
    plot(a_1_2_std);
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'centralized-urgency-then-cost';
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
            plot(a_lim_mem_u_std{i_lim_mem});
            axes = gca;
            axis tight;
            axes.Title.Interpreter = 'latex';
            axes.Title.String = ['centralized-urgency-then-cost-mem-', int2str(param.lim_mem_steps(i_lim_mem))];
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
        
        title = sgtitle('Standardized accumulated costs for limited memory policies');
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
        subplot(2,2,1);
        plot(a_bid_1_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-1-always';
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
        plot(a_bid_1_u_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-1-if-urgent';
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
        plot(a_bid_all_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-all-always';
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
        plot(a_bid_all_u_std);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = 'bid-all-if-urgent';
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
        
        title = sgtitle(['$k_{max}$ = ', num2str(param.k_max, '%02d'), ' $k_{ave}$ = ', num2str(param.k_ave, '%02d'), ' standardized accumulated costs for heuristic karma policies']);
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
                axes.Title.String = ['$\alpha$ = ', num2str(param.alpha(i_subplot), '%.2f')];
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
        
        title = sgtitle(['$k_{max}$ = ', num2str(param.k_max, '%02d'), ' $k_{ave}$ = ', num2str(param.k_ave, '%02d'), ' standardized accumulated costs for karma policies']);
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
    plot(acorr_tau, a_1_acorr, 'LineWidth', 2);
    plot(acorr_tau, a_2_acorr, 'LineWidth', 2);
    plot(acorr_tau, a_1_2_acorr, 'LineWidth', 2);
    if control.lim_mem_policies
        for i_lim_mem = 1 : param.num_lim_mem_steps
            plot(acorr_tau, a_lim_mem_u_acorr{i_lim_mem}, ':');
        end
    end
    if control.karma_heuristic_policies
        plot(acorr_tau, a_bid_1_acorr, '-.');
        plot(acorr_tau, a_bid_1_u_acorr, '-.');
        plot(acorr_tau, a_bid_all_acorr, '-.');
        plot(acorr_tau, a_bid_all_u_acorr, '-.');
    end
    if control.karma_ne_policies
        for i_alpha = 1 : param.num_alpha
            plot(acorr_tau, a_ne_acorr{i_alpha}, '--');
        end
    end
    if control.karma_sw_policy
        plot(acorr_tau, a_sw_acorr, '--', 'LineWidth', 2);
    end
    axis tight;
    axes = gca;
    yl = ylim(axes);
    stem(0, a_rand_acorr(acorr_tau == 0));
    ylim(axes, yl);
    axes.Title.Interpreter = 'latex';
    axes.Title.String = ['$k_{max}$ = ', num2str(param.k_max, '%02d'), ' $k_{ave}$ = ', num2str(param.k_ave, '%02d'), ' autocorrelation of accumulated costs'];
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