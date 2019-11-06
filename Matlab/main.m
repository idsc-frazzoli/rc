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
param.num_iter = 10000;

% Minimum karma level
param.k_min = 0;

% Maximum karma level
param.k_max = 12;

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
% Cost for bid 1 always policy
c_bid_1 = zeros(param.num_iter, param.N);
% Cost for bid 1 if urgent policy
c_bid_1_u = zeros(param.num_iter, param.N);

% Karma matrices for different polices. Initialized uniformly randomly
% between k_min & k_max. Same initialization for all policies
karma_init = round(rand(1, param.N) * (param.k_max - param.k_min)) + param.k_min;
% Karma for bid 1 always policy
k_bid_1 = zeros(param.num_iter, param.N);
k_bid_1(1,:) = karma_init;
% Karma for bid 1 if urgent policy
k_bid_1_u = zeros(param.num_iter, param.N);
k_bid_1_u(1,:) = karma_init;

% Number of times each agent was in an intersection, as a cumulative sum
num_inter = zeros(param.num_iter, param.N);

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
    if t > 1
        num_inter(t,:) = num_inter(t-1,:);
        num_inter(t,I) = num_inter(t,I) + 1;
    else
        num_inter(t,I) = 1;
    end
    
    % Pick urgency in {0,U} uniformly at random
    u = round(rand(1, param.I_size)) * param.U;
    
    %% Random policy
    % Choose an agent to pass uniformly at random
    p = I(ceil(rand(1) * length(I)));
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities
    c_rand(t,I) = u;
    c_rand(t,p) = 0;
    
    %% Centralized policy 1 - minimize W1, coin-flip if tie
    % Find agent(s) with max urgency, which are candidates for passing
    u_temp = u;
    [max_v, max_i] = max(u_temp);
    p_max_u = I(max_i);
    u_temp(max_i) = -realmax;
    [next_max_v, max_i] = max(u_temp);
    while next_max_v == max_v
        p_max_u = [p_max_u, I(max_i)];
        u_temp(max_i) = -realmax;
        [next_max_v, max_i] = max(u_temp);
    end
    
    % Now choose an agent uniformly at random if there are multiple.
    % Note that index 1 results if there is only one
    p = p_max_u(ceil(rand(1) * length(p_max_u)));
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities
    c_1(t,I) = u;
    c_1(t,p) = 0;
    
    %% Centralized policy 2 - minimize W2, coin-flip if tie
    % Find agent(s) with max accumulated cost plus urgency, which are
    % candidates for passing
    a_u_temp = sum(c_2(1:t,I)) + u;
    [max_v, max_i] = max(a_u_temp);
    p = I(max_i);
    a_u_temp(max_i) = -realmax;
    [next_max_v, max_i] = max(a_u_temp);
    while next_max_v == max_v
        p = [p, I(max_i)];
        a_u_temp(max_i) = -realmax;
        [next_max_v, max_i] = max(a_u_temp);
    end
    
    % Now choose an agent uniformly at random if there are multiple.
    % Note that index 1 results if there is only one
    p = p(ceil(rand(1) * length(p)));
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities
    c_2(t,I) = u;
    c_2(t,p) = 0;
    
    %% Centralized policy 1_2 - minimize W1, choose W2 minimizer on tie
    % Agent(s) with max urgency, which are candidates for passing, were
    % already found in first step of centralized policy 1
    if length(p_max_u) > 1
        p_ind = [];
        for i = 1 : length(p_max_u)
            p_ind = [p_ind, find(I == p_max_u(i))];
        end
        a_u_temp = sum(c_1_2(1:t,p_max_u)) + u(p_ind);
        [max_v, max_i] = max(a_u_temp);
        p = p_max_u(max_i);
        a_u_temp(max_i) = -realmax;
        [next_max_v, max_i] = max(a_u_temp);
        while next_max_v == max_v
            p = [p, p_max_u(max_i)];
            a_u_temp(max_i) = -realmax;
            [next_max_v, max_i] = max(a_u_temp);
        end

        % Now choose an agent uniformly at random if there are multiple.
        % Note that index 1 results if there is only one
        p = p(ceil(rand(1) * length(p)));
    else
        p = p_max_u;
    end
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities
    c_1_2(t,I) = u;
    c_1_2(t,p) = 0;
    
    %% Bid 1 always policy
    % Agents simply bid all their karma, less the minimum allowed level (if
    % applicable)
    k = k_bid_1(t,I) - param.k_min;
    
    % Determine winning bid
    k_temp = k;
    [max_v, max_i] = max(k_temp);
    win_i = max_i;
    k_temp(max_i) = -realmax;
    [next_max_v, max_i] = max(k_temp);
    while next_max_v == max_v
        win_i = [win_i, max_i];
        k_temp(max_i) = -realmax;
        [next_max_v, max_i] = max(k_temp);
    end
    
    % Now choose an agent uniformly at random if there are multiple.
    % Note that index 1 results if there is only one
    win_i = win_i(ceil(rand(1) * length(win_i)));
    p = I(win_i);
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities
    c_bid_1(t,I) = u;
    c_bid_1(t,p) = 0;
    
    % Determine agents that will be delayed. They will be getting karma
    d = I;
    d(win_i) = [];
    
    % Update karma values
    if t < param.num_iter
        % Determine karma payment from passing agent to each of delayed
        % agents
        % Candidate karma paid by passing agent
        k_p = k(win_i);
        % Distribute karma evenly over delayed agents. If an agent will max
        % out their karma, tough luck!
        k_d = zeros(1, length(d));
        for i = 1 : length(d)
            k_d(i) = min([floor(k_p / length(d)), param.k_max - k_bid_1(t,d(i))]);
        end
        % Sum back the total karma distributed, which takes into account
        % delayed agents for which karma will saturate. This is the final
        % total paid by passing agent
        k_p = sum(k_d);
        
        % Initialize next karma values to current karma values
        k_bid_1(t+1,:) = k_bid_1(t,:);
        
        % Agent that passes loses karma
        k_bid_1(t+1,p) = k_bid_1(t+1,p) - k_p;
        
        % Agents that are delayed get karma
        k_bid_1(t+1,d) = k_bid_1(t+1,d) + k_d;
    end
    
    %% Bid 1 if urgent policy
    % Agents bid all their karma, less the minimum allowed level (if
    % applicable), if they are urgent
    k = k_bid_1_u(t,I) - param.k_min;
    k(u == 0) = 0;
    
    % Determine winning bid
    k_temp = k;
    [max_v, max_i] = max(k_temp);
    win_i = max_i;
    k_temp(max_i) = -realmax;
    [next_max_v, max_i] = max(k_temp);
    while next_max_v == max_v
        win_i = [win_i, max_i];
        k_temp(max_i) = -realmax;
        [next_max_v, max_i] = max(k_temp);
    end
    
    % Now choose an agent uniformly at random if there are multiple.
    % Note that index 1 results if there is only one
    win_i = win_i(ceil(rand(1) * length(win_i)));
    p = I(win_i);
    
    % Agent that passed incurs no cost. All other agents incur cost that
    % equals their utilities
    c_bid_1_u(t,I) = u;
    c_bid_1_u(t,p) = 0;
    
    % Determine agents that will be delayed. They will be getting karma
    d = I;
    d(win_i) = [];
    
    % Update karma values
    if t < param.num_iter
        % Determine karma payment from passing agent to each of delayed
        % agents
        % Candidate karma paid by passing agent
        k_p = k(win_i);
        % Distribute karma evenly over delayed agents. If an agent will max
        % out their karma, tough luck!
        k_d = zeros(1, length(d));
        for i = 1 : length(d)
            k_d(i) = min([floor(k_p / length(d)), param.k_max - k_bid_1_u(t,d(i))]);
        end
        % Sum back the total karma distributed, which takes into account
        % delayed agents for which karma will saturate. This is the final
        % total paid by passing agent
        k_p = sum(k_d);
        
        % Initialize next karma values to current karma values
        k_bid_1_u(t+1,:) = k_bid_1_u(t,:);
        
        % Agent that passes loses karma
        k_bid_1_u(t+1,p) = k_bid_1_u(t+1,p) - k_p;
        
        % Agents that are delayed get karma
        k_bid_1_u(t+1,d) = k_bid_1_u(t+1,d) + k_d;
    end
end

%% Perfromance measures
% Cumulative costs per agent at each time step
c_rand_cumsum = cumsum(c_rand);
c_1_cumsum = cumsum(c_1);
c_2_cumsum = cumsum(c_2);
c_1_2_cumsum = cumsum(c_1_2);
c_bid_1_cumsum = cumsum(c_bid_1);
c_bid_1_u_cumsum = cumsum(c_bid_1_u);

% Cumulative costs per agent at each time step, normalized by their
% respective number of interactions
% Note that nan can result if agent(s) were never in an intersection, which
% is handled later by using 'nan---' functions (which ignore nan's)
c_rand_cumsum_norm = c_rand_cumsum ./ num_inter;
c_1_cumsum_norm = c_1_cumsum ./ num_inter;
c_2_cumsum_norm = c_2_cumsum ./ num_inter;
c_1_2_cumsum_norm = c_1_2_cumsum ./ num_inter;
c_bid_1_cumsum_norm = c_bid_1_cumsum ./ num_inter;
c_bid_1_u_cumsum_norm = c_bid_1_u_cumsum ./ num_inter;

% Total costs per time step
c_rand_tot_t = sum(c_rand, 2);
c_1_tot_t = sum(c_1, 2);
c_2_tot_t = sum(c_2, 2);
c_1_2_tot_t = sum(c_1_2, 2);
c_bid_1_tot_t = sum(c_bid_1, 2);
c_bid_1_u_tot_t = sum(c_bid_1_u, 2);

% Inefficiency vs. time
W1_rand = nanmean(c_rand_cumsum, 2);
W1_1 = nanmean(c_1_cumsum, 2);
W1_2 = nanmean(c_2_cumsum, 2);
W1_1_2 = nanmean(c_1_2_cumsum, 2);
W1_bid_1 = nanmean(c_bid_1_cumsum, 2);
W1_bid_1_u = nanmean(c_bid_1_u_cumsum, 2);

% Normalized inefficiency vs. time
W1_rand_norm = nanmean(c_rand_cumsum_norm, 2);
W1_1_norm = nanmean(c_1_cumsum_norm, 2);
W1_2_norm = nanmean(c_2_cumsum_norm, 2);
W1_1_2_norm = nanmean(c_1_2_cumsum_norm, 2);
W1_bid_1_norm = nanmean(c_bid_1_cumsum_norm, 2);
W1_bid_1_u_norm = nanmean(c_bid_1_u_cumsum_norm, 2);

% Unfairness vs. time
W2_rand = nanvar(c_rand_cumsum, [], 2);
W2_1 = nanvar(c_1_cumsum, [], 2);
W2_2 = nanvar(c_2_cumsum, [], 2);
W2_1_2 = nanvar(c_1_2_cumsum, [], 2);
W2_bid_1 = nanvar(c_bid_1_cumsum, [], 2);
W2_bid_1_u = nanvar(c_bid_1_u_cumsum, [], 2);

% Normalized unfairness vs. time
W2_rand_norm = nanvar(c_rand_cumsum_norm, [], 2);
W2_1_norm = nanvar(c_1_cumsum_norm, [], 2);
W2_2_norm = nanvar(c_2_cumsum_norm, [], 2);
W2_1_2_norm = nanvar(c_1_2_cumsum_norm, [], 2);
W2_bid_1_norm = nanvar(c_bid_1_cumsum_norm, [], 2);
W2_bid_1_u_norm = nanvar(c_bid_1_u_cumsum_norm, [], 2);

%% Scatter plot - Inefficiency vs Unfairness
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
% Non-normalized
subplot(2,1,1);
p = plot(W1_rand(end), W2_rand(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = "baseline-random";
hold on;
p = plot(W1_1(end), W2_1(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "centralized-urgency"];
p = plot(W1_2(end), W2_2(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "centralized-cost"];
p = plot(W1_1_2(end), W2_1_2(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "centralized-urgency-then-cost"];
p = plot(W1_bid_1(end), W2_bid_1(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "bid1-always"];
p = plot(W1_bid_1_u(end), W2_bid_1_u(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "bid1-if-urgent"];
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
axes.YLabel.String = 'Unfairness (variance of cost)';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.Interpreter = 'latex';
lgd.FontSize = 14;
lgd.Location = 'bestoutside';
% Normalized
subplot(2,1,2);
p = plot(W1_rand_norm(end), W2_rand_norm(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
hold on;
p = plot(W1_1_norm(end), W2_1_norm(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
p = plot(W1_2_norm(end), W2_2_norm(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
p = plot(W1_1_2_norm(end), W2_1_2_norm(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
p = plot(W1_bid_1_norm(end), W2_bid_1_norm(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
p = plot(W1_bid_1_u_norm(end), W2_bid_1_u_norm(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
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
axes.XLabel.String = 'Normalized inefficiency (mean of normalized cost)';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized unfairness (variance of normalized cost)';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.Interpreter = 'latex';
lgd.FontSize = 14;
lgd.Location = 'bestoutside';

%% Cumulative cost plot - Gives indication on how fast variance grows
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
subplot(4,2,1);
plot(c_rand_cumsum);
hold on;
plot(W1_rand, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'baseline-random';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,2);
plot(c_1_cumsum);
hold on;
plot(W1_1, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,3);
plot(c_2_cumsum);
hold on;
plot(W1_2, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-cost';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,4);
plot(c_1_2_cumsum);
hold on;
plot(W1_1_2, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency-then-cost';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,5);
plot(c_bid_1_cumsum);
hold on;
plot(W1_bid_1, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid1-always';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,6);
plot(c_bid_1_u_cumsum);
hold on;
plot(W1_bid_1_u, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid1-if-urgent';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,[7 8]);
plot(W2_rand);
hold on;
plot(W2_1);
plot(W2_2);
plot(W2_1_2);
plot(W2_bid_1);
plot(W2_bid_1_u);
axes = gca;
axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'Unfairness vs. time';
% axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Unfairness';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.Interpreter = 'latex';
lgd.FontSize = 12;
lgd.Location = 'bestoutside';

%% Normalized cumulative cost plot
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
subplot(4,2,1);
plot(c_rand_cumsum_norm);
hold on;
plot(W1_rand_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'baseline-random';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,2);
plot(c_1_cumsum_norm);
hold on;
plot(W1_1_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,3);
plot(c_2_cumsum_norm);
hold on;
plot(W1_2_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-cost';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,4);
plot(c_1_2_cumsum_norm);
hold on;
plot(W1_1_2_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency-then-cost';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,5);
plot(c_bid_1_cumsum_norm);
hold on;
plot(W1_bid_1_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid1-always';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,6);
plot(c_bid_1_u_cumsum_norm);
hold on;
plot(W1_bid_1_u_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid1-if-urgent';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 14;
subplot(4,2,[7 8]);
plot(W2_rand_norm);
hold on;
plot(W2_1_norm);
plot(W2_2_norm);
plot(W2_1_2_norm);
plot(W2_bid_1_norm);
plot(W2_bid_1_u_norm);
axes = gca;
axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'Normalized unfairness vs. time';
% axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized unfairness';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.Interpreter = 'latex';
lgd.FontSize = 12;
lgd.Location = 'bestoutside';

%% Number of interactions is a random walk!!
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, default_height/2, default_width, default_height];
plot(num_inter);
hold on;
plot(mean(num_inter, 2), 'Linewidth', 10);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'Number of interactions per agent';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Number of interactions';
axes.YLabel.FontSize = 14;

%% Inform user when done
fprintf('DONE\n\n');

%% Helper functions
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

function y_semi_tight(ax, scale)
    axis tight; % Set axis tight
    yl = ylim(ax); % Get tight axis limits
    range = yl(2) - yl(1); % Get tight axis range
    sc_range = scale * range; % Scale range
    yl(1) = yl(1) - (sc_range - range) / 2; % New ymin
    yl(2) = yl(1) + sc_range; % New ymax
    ylim(ax, yl);
end