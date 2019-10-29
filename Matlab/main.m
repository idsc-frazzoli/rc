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

%% Fix rung for randomization
rng(0);

%% Parameters
% Population size
param.N = 200;

% Discrete urgency level
param.U = 3;

% Number of agents in one intersection
param.I_size = 2;

% Number of iterations to run simulation
param.num_iter = 1000;

%% Simulation initialization
% Cost matrices for different policies
% Row => Time step
% Col => Agent
% Cost for baseline policy - random
c_rand = zeros(param.num_iter, param.N);
% Cost for centralized policy 1 - centralized urgency
c_1 = zeros(param.num_iter, param.N);
% Cost for centralized policy 2 - centralized cost
c_2 = zeros(param.num_iter, param.N);
% Cost for centralized policy 1_2 - centralized urgency then cost
c_1_2 = zeros(param.num_iter, param.N);

% Number of times each agent was in an intersection
num_inter = zeros(1, param.N);

%% Simulation run
% Convention:   p := agent that passes
%               d := agent(s) that are delayed
for t = 1 : param.num_iter
    % Tell user where we are
    fprintf('t = %d\n', t);
    
    % Pick vehicles i & j uniformly at random
    % Make sure vehicles are unique!
    I = ceil(rand(1, param.I_size) * param.N);
    while length(I) ~= length(unique(I))
        I = ceil(rand(1, param.I_size) * param.N);
    end
    
    % Increment number of intersections for picked agents
    num_inter(I) = num_inter(I) + 1;
    
    % Pick urgency in {0,U} uniformly at random
    u = round(rand(1, param.I_size)) * param.U;
    
    %% Random policy
    % Choose an agent to pass uniformly at random
    p_rand = I(ceil(rand(1) * length(I)));
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities. Here the passing agent is 'flicked' out
    i = find(I == p_rand);
    d_rand = I;
    u_temp = u;
    d_rand(i) = [];
    u_temp(i) = [];
    
    % Update cost matrix for delayed agents
    for i = 1 : length(d_rand)
        c_rand(t,d_rand(i)) = u_temp(i);
    end
    
    %% Centralized policy 1 - minimize W1, coin-flip if tie
    % Find agent(s) with max urgency, which are candidates for passing
    u_temp = u;
    [max_v, max_i] = max(u_temp);
    p = I(max_i);
    u_temp(max_i) = -realmax;
    [next_max_v, max_i] = max(u_temp);
    while next_max_v == max_v
        p = [p, I(max_i)];
        u_temp(max_i) = -realmax;
        [next_max_v, max_i] = max(u_temp);
    end
    
    % Now choose an agent uniformly at random if there are multiple.
    % Note that index 1 results if there is only one
    p_1 = p(ceil(rand(1) * length(p)));
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities. Here the passing agent is 'flicked' out
    i = find(I == p_1);
    d_1 = I;
    u_temp = u;
    d_1(i) = [];
    u_temp(i) = [];
    
    % Update cost matrix for delayed agents
    for i = 1 : length(d_1)
        c_1(t,d_1(i)) = u_temp(i);
    end
    
    %% Centralized policy 2 - minimize W2, coin-flip if tie
    % Find agent(s) with max accumulated cost plus urgency, which are
    % candidates for passing
    a_u_temp = sum(c_2(1:t,I)) + u;
    [max_v, max_i] = max(a_u_temp);
    p_2 = I(max_i);
    a_u_temp(max_i) = -realmax;
    [next_max_v, max_i] = max(a_u_temp);
    while next_max_v == max_v
        p_2 = [p_2, I(max_i)];
        a_u_temp(max_i) = -realmax;
        [next_max_v, max_i] = max(a_u_temp);
    end
    
    % Now choose an agent uniformly at random if there are multiple.
    % Note that index 1 results if there is only one
    p_2 = p_2(ceil(rand(1) * length(p_2)));
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities. Here the passing agent is 'flicked' out
    i = find(I == p_2);
    d_2 = I;
    u_temp = u;
    d_2(i) = [];
    u_temp(i) = [];
    
    % Update cost matrix for delayed agents
    for i = 1 : length(d_2)
        c_2(t,d_2(i)) = u_temp(i);
    end
    
    %% Centralized policy 1_2 - minimize W1, choose W2 minimizer on tie
    % Agent(s) with max urgency, which are candidates for passing, were
    % already found in first step of centralized policy 1
    if length(p) > 1
        p_ind = [];
        for i = 1 : length(p)
            p_ind = [p_ind, find(I == p(i))];
        end
        a_u_temp = sum(c_1_2(1:t,p)) + u(p_ind);
        [max_v, max_i] = max(a_u_temp);
        p_1_2 = p(max_i);
        a_u_temp(max_i) = -realmax;
        [next_max_v, max_i] = max(a_u_temp);
        while next_max_v == max_v
            p_1_2 = [p_1_2, p(max_i)];
            a_u_temp(max_i) = -realmax;
            [next_max_v, max_i] = max(a_u_temp);
        end

        % Now choose an agent uniformly at random if there are multiple.
        % Note that index 1 results if there is only one
        p_1_2 = p_1_2(ceil(rand(1) * length(p_1_2)));
    else
        p_1_2 = p;
    end
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities. Here the passing agent is 'flicked' out
    i = find(I == p_1_2);
    d_1_2 = I;
    u_temp = u;
    d_1_2(i) = [];
    u_temp(i) = [];
    
    % Update cost matrix for delayed agents
    for i = 1 : length(d_1_2)
        c_1_2(t,d_1_2(i)) = u_temp(i);
    end
end

%% Perfromance measures
% Total costs per agent, normalized by their respective number of
% intersections
% If agent was never in an intersection (results in nan) simply ignore
c_rand_tot = sum(c_rand) ./ num_inter;
c_rand_tot(isnan(c_rand_tot)) = [];
c_1_tot = sum(c_1) ./ num_inter;
c_1_tot(isnan(c_1_tot)) = [];
c_2_tot = sum(c_2) ./ num_inter;
c_2_tot(isnan(c_2_tot)) = [];
c_1_2_tot = sum(c_1_2) ./ num_inter;
c_1_2_tot(isnan(c_1_2_tot)) = [];

% Total costs per time step
c_rand_tot_t = sum(c_rand, 2);
c_1_tot_t = sum(c_1, 2);
c_2_tot_t = sum(c_2, 2);
c_1_2_tot_t = sum(c_1_2, 2);

% Inefficiency
W1_rand = mean(c_rand_tot);
W1_1 = mean(c_1_tot);
W1_2 = mean(c_2_tot);
W1_1_2 = mean(c_1_2_tot);

% Unfairness
W2_rand = std(c_rand_tot);
W2_1 = std(c_1_tot);
W2_2 = std(c_2_tot);
W2_1_2 = std(c_1_2_tot);

%% Scatter plot - Inefficiency vs Unfairness
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, screenheight/4, default_width, default_height];
p = plot(W1_rand, W2_rand,...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = "baseline-random";
hold on;
p = plot(W1_1, W2_1,...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "centralized-urgency"];
p = plot(W1_2, W2_2,...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "centralized-cost"];
p = plot(W1_1_2, W2_1_2,...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "centralized-urgency-then-cost"];
axes = gca;
axis_semi_tight(axes, 1.2);
axes.Title.Interpreter = 'latex';
axes.Title.String = 'Performance Comparison';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Inefficiency (mean of cost)';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Unfairness (std-dev of cost)';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.Interpreter = 'latex';
lgd.FontSize = 14;
lgd.Location = 'bestoutside';

%% Inform user when done
fprintf('DONE\n\n');

function axis_semi_tight(ax, scale)
    axis tight; % Set axis tight
    % x limits
    xl = xlim(ax); % Get tight axis limits
    range = xl(2) - xl(1); % Get tight axis range
    sc_range = scale * range; % Scale range
    xl(1) = xl(1) - (sc_range - range) / 2; % New xmin
    xl(2) = xl(1) + sc_range; % New xmax
    xlim(ax, xl);
    % y limits
    yl = ylim(ax); % Get tight axis limits
    range = yl(2) - yl(1); % Get tight axis range
    sc_range = scale * range; % Scale range
    yl(1) = yl(1) - (sc_range - range) / 2; % New ymin
    yl(2) = yl(1) + sc_range; % New ymax
    ylim(ax, yl);
end