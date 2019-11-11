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

% Number of agents that pass
param.num_p = 1;

% Number of agents that get delayed
param.num_d = param.I_size - param.num_p;

% Number of iterations to run simulation
param.num_iter = 10000;

% Minimum karma level
param.k_min = 0;

% Maximum karma level
param.k_max = 12;

% Future discount factor
param.alpha = 0.5;

% Initial policy which maps
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
% Cost for bid all always policy
c_bid_all = zeros(param.num_iter, param.N);
% Cost for bid all if urgent policy
c_bid_all_u = zeros(param.num_iter, param.N);

% Karma matrices for different polices. Initialized uniformly randomly
% between k_min & k_max. Same initialization for all policies
karma_init = round(rand(1, param.N) * (param.k_max - param.k_min)) + param.k_min;
% Karma for bid 1 always policy
k_bid_1 = zeros(param.num_iter, param.N);
k_bid_1(1,:) = karma_init;
% Karma for bid 1 if urgent policy
k_bid_1_u = zeros(param.num_iter, param.N);
k_bid_1_u(1,:) = karma_init;
% Karma for bid all always policy
k_bid_all = zeros(param.num_iter, param.N);
k_bid_all(1,:) = karma_init;
% Karma for bid all if urgent policy
k_bid_all_u = zeros(param.num_iter, param.N);
k_bid_all_u(1,:) = karma_init;

% Number of times each agent was in an intersection, as a cumulative sum
num_inter = zeros(param.num_iter, param.N);

%% Karma Nash equilibrium policy calculation
% Vector of all karma values
k = param.k_min : 1 : param.k_max;
num_k = length(k);

% Policy function, parametrized as a (num_k x num_k) matrix. Entry (i,j)
% denotes probability of transmitting message i when karma level is j. Note
% that columns must sum to 1
% Initialize to the identity, which is equivalent to bid-all-if-urgent
% (alpha = 0)
pie = eye(num_k);

% Stationary distribution, parametrized as a vector with num_k cols.
% Note that it sums to 1
% Initialize to uniform distribution 
D = 1 / num_k * ones(1, num_k);

% Utility function, parametrized as a vector with num_k cols
% Initialize to zero
theta = zeros(1, num_k);

% k_next cell of matrices. Each matrix corresponds to a current karma level
% Matrix describes agent 1's next karma in terms of agent 1 (rows) and
% agent 2 (cols) bids. Note that agent is only allowed to bid up to current
% karma level, which is why matrices in cell have different number of rows
k_next = cell(1, num_k);
for i_1 = 1 : num_k
    k_curr = k(i_1);
    k_next{i_1} = zeros(i_1, num_k);
    for i_2 = 1 : i_1
        for i_3 = 1 : num_k
            if k(i_2) < k(i_3)  % Agent i receives karma, up to max level
                k_next{i_1}(i_2,i_3) = min([k_curr + k(i_3), param.k_max]);
            elseif k(i_2) > k(i_3) % Agent i pays karma
                k_next{i_1}(i_2,i_3) = k_curr - k(i_2);
            else                    % Agent
                k_next{i_1}(i_2,i_3) = k_curr;
            end
        end
    end
end

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
    
    % Agents incur cost equal to their urgency, except passing agent
    c_rand(t,I) = u;
    c_rand(t,p) = 0;
    
    %% Centralized policy 1 - minimize W1, coin-flip if tie
    % Find agent(s) with max urgency, which are candidates for passing
    [~, p_i] = multi_maxes(u);
    p_max_u = I(p_i);
    
    % Now choose an agent uniformly at random if there are multiple.
    num_max_u = length(p_max_u);
    if num_max_u > 1
        p = p_max_u(ceil(rand(1) * num_max_u));
    else
        p = p_max_u;
    end
    
    % Agents incur cost equal to their urgency, except passing agent
    c_1(t,I) = u;
    c_1(t,p) = 0;
    
    %% Centralized policy 2 - minimize W2, coin-flip if tie
    % Agent with maximum accumulated cost (counting current urgency) passes
    a_u = sum(c_2(1:t,I)) + u;
    [~, p_i] = multi_max(a_u);
    p = I(p_i);
    
    % Agents incur cost equal to their urgency, except passing agent
    c_2(t,I) = u;
    c_2(t,p) = 0;
    
    %% Centralized policy 1_2 - minimize W1, choose W2 minimizer on tie
    % Agent(s) with max urgency, which are candidates for passing, were
    % already found in first step of centralized policy 1
    % If there are multiple agents with max urgency, pick on based on
    % accumulated cost like in centralized policy 2
    if num_max_u > 1
        p_ind = zeros(1, num_max_u);
        for i = 1 : num_max_u
            p_ind(i) = find(I == p_max_u(i));
        end
        a_u = sum(c_1_2(1:t,p_max_u)) + u(p_ind);
        [~, p_i] = multi_max(a_u);
        p = p_max_u(p_i);
    else
        p = p_max_u;
    end
    
    % Agents incur cost equal to their urgency, except passing agent
    c_1_2(t,I) = u;
    c_1_2(t,p) = 0;
    
    %% Bid 1 always policy
    % Agents simply bid 1, if they have it
    m = min([ones(1, param.I_size); k_bid_1(t,I) - param.k_min]);
    
    % Agent bidding max karma passes and pays karma bidded
    [m_p, p_i] = multi_max(m);
    p = I(p_i);
    
    % Agents incur cost equal to their urgency, except passing agent
    c_bid_1(t,I) = u;
    c_bid_1(t,p) = 0;
    
    % Get delayed agents. They will be getting karma
    d = get_d(I, p_i);
    
    % Update karma
    if t < param.num_iter
        [k_p, k_d] = get_karma_payments(m_p, d, k_bid_1(t,:), param);
        k_bid_1(t+1,:) = k_bid_1(t,:);
        k_bid_1(t+1,p) = k_bid_1(t+1,p) - k_p;
        k_bid_1(t+1,d) = k_bid_1(t+1,d) + k_d;
    end
    
    %% Bid 1 if urgent policy
    % Agents bid 1, if they have it and they are urgent
    m = min([ones(1, param.I_size); k_bid_1_u(t,I) - param.k_min]);
    m(u == 0) = 0;
    
    % Agent bidding max karma passes and pays karma bidded
    [m_p, p_i] = multi_max(m);
    p = I(p_i);
    
    % Agents incur cost equal to their urgency, except passing agent
    c_bid_1_u(t,I) = u;
    c_bid_1_u(t,p) = 0;
    
    % Get delayed agents. They will be getting karma
    d = get_d(I, p_i);
    
    % Update karma
    if t < param.num_iter
        [k_p, k_d] = get_karma_payments(m_p, d, k_bid_1_u(t,:), param);
        k_bid_1_u(t+1,:) = k_bid_1_u(t,:);
        k_bid_1_u(t+1,p) = k_bid_1_u(t+1,p) - k_p;
        k_bid_1_u(t+1,d) = k_bid_1_u(t+1,d) + k_d;
    end
    
    %% Bid all always policy
    % Agents simply bid all their karma, less the minimum allowed level (if
    % applicable)
    m = k_bid_all(t,I) - param.k_min;
    
    % Agent bidding max karma passes and pays karma bidded
    [m_p, p_i] = multi_max(m);
    p = I(p_i);
    
    % Agents incur cost equal to their urgency, except passing agent
    c_bid_all(t,I) = u;
    c_bid_all(t,p) = 0;
    
    % Get delayed agents. They will be getting karma
    d = get_d(I, p_i);
    
    % Update karma
    if t < param.num_iter
        [k_p, k_d] = get_karma_payments(m_p, d, k_bid_all(t,:), param);
        k_bid_all(t+1,:) = k_bid_all(t,:);
        k_bid_all(t+1,p) = k_bid_all(t+1,p) - k_p;
        k_bid_all(t+1,d) = k_bid_all(t+1,d) + k_d;
    end
    
    %% Bid all if urgent policy
    % Agents bid all their karma, less the minimum allowed level (if
    % applicable), if they are urgent
    m = k_bid_all_u(t,I) - param.k_min;
    m(u == 0) = 0;
    
    % Agent bidding max karma passes and pays karma bidded
    [m_p, p_i] = multi_max(m);
    p = I(p_i);
    
    % Agents incur cost equal to their urgency, except passing agent
    c_bid_all_u(t,I) = u;
    c_bid_all_u(t,p) = 0;
    
    % Get delayed agents. They will be getting karma
    d = get_d(I, p_i);
    
    % Update karma
    if t < param.num_iter
        [k_p, k_d] = get_karma_payments(m_p, d, k_bid_all_u(t,:), param);
        k_bid_all_u(t+1,:) = k_bid_all_u(t,:);
        k_bid_all_u(t+1,p) = k_bid_all_u(t+1,p) - k_p;
        k_bid_all_u(t+1,d) = k_bid_all_u(t+1,d) + k_d;
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
c_bid_all_cumsum = cumsum(c_bid_all);
c_bid_all_u_cumsum = cumsum(c_bid_all_u);

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
c_bid_all_cumsum_norm = c_bid_all_cumsum ./ num_inter;
c_bid_all_u_cumsum_norm = c_bid_all_u_cumsum ./ num_inter;

% Total costs per time step
c_rand_tot_t = sum(c_rand, 2);
c_1_tot_t = sum(c_1, 2);
c_2_tot_t = sum(c_2, 2);
c_1_2_tot_t = sum(c_1_2, 2);
c_bid_1_tot_t = sum(c_bid_1, 2);
c_bid_1_u_tot_t = sum(c_bid_1_u, 2);
c_bid_all_tot_t = sum(c_bid_all, 2);
c_bid_all_u_tot_t = sum(c_bid_all_u, 2);

% Inefficiency vs. time
W1_rand = nanmean(c_rand_cumsum, 2);
W1_1 = nanmean(c_1_cumsum, 2);
W1_2 = nanmean(c_2_cumsum, 2);
W1_1_2 = nanmean(c_1_2_cumsum, 2);
W1_bid_1 = nanmean(c_bid_1_cumsum, 2);
W1_bid_1_u = nanmean(c_bid_1_u_cumsum, 2);
W1_bid_all = nanmean(c_bid_all_cumsum, 2);
W1_bid_all_u = nanmean(c_bid_all_u_cumsum, 2);

% Normalized inefficiency vs. time
W1_rand_norm = nanmean(c_rand_cumsum_norm, 2);
W1_1_norm = nanmean(c_1_cumsum_norm, 2);
W1_2_norm = nanmean(c_2_cumsum_norm, 2);
W1_1_2_norm = nanmean(c_1_2_cumsum_norm, 2);
W1_bid_1_norm = nanmean(c_bid_1_cumsum_norm, 2);
W1_bid_1_u_norm = nanmean(c_bid_1_u_cumsum_norm, 2);
W1_bid_all_norm = nanmean(c_bid_all_cumsum_norm, 2);
W1_bid_all_u_norm = nanmean(c_bid_all_u_cumsum_norm, 2);

% Unfairness vs. time
W2_rand = nanvar(c_rand_cumsum, [], 2);
W2_1 = nanvar(c_1_cumsum, [], 2);
W2_2 = nanvar(c_2_cumsum, [], 2);
W2_1_2 = nanvar(c_1_2_cumsum, [], 2);
W2_bid_1 = nanvar(c_bid_1_cumsum, [], 2);
W2_bid_1_u = nanvar(c_bid_1_u_cumsum, [], 2);
W2_bid_all = nanvar(c_bid_all_cumsum, [], 2);
W2_bid_all_u = nanvar(c_bid_all_u_cumsum, [], 2);

% Normalized unfairness vs. time
W2_rand_norm = nanvar(c_rand_cumsum_norm, [], 2);
W2_1_norm = nanvar(c_1_cumsum_norm, [], 2);
W2_2_norm = nanvar(c_2_cumsum_norm, [], 2);
W2_1_2_norm = nanvar(c_1_2_cumsum_norm, [], 2);
W2_bid_1_norm = nanvar(c_bid_1_cumsum_norm, [], 2);
W2_bid_1_u_norm = nanvar(c_bid_1_u_cumsum_norm, [], 2);
W2_bid_all_norm = nanvar(c_bid_all_cumsum_norm, [], 2);
W2_bid_all_u_norm = nanvar(c_bid_all_u_cumsum_norm, [], 2);

%% Correlation of accumulated cost over time
% Provides indication on how well population cost 'mixes' with time

% Accumulated costs must be standardized first for correlation to be
% meaningful, i.e. brought to zero mean and unit std-dev
c_rand_cumsum_std = (c_rand_cumsum - W1_rand) ./ sqrt(W2_rand);
c_1_cumsum_std = (c_1_cumsum - W1_1) ./ sqrt(W2_1);
c_2_cumsum_std = (c_2_cumsum - W1_2) ./ sqrt(W2_2);
c_1_2_cumsum_std = (c_1_2_cumsum - W1_1_2) ./ sqrt(W2_1_2);
c_bid_1_cumsum_std = (c_bid_1_cumsum - W1_bid_1) ./ sqrt(W2_bid_1);
c_bid_1_u_cumsum_std = (c_bid_1_u_cumsum - W1_bid_1_u) ./ sqrt(W2_bid_1_u);
c_bid_all_cumsum_std = (c_bid_all_cumsum - W1_bid_all) ./ sqrt(W2_bid_all);
c_bid_all_u_cumsum_std = (c_bid_all_u_cumsum - W1_bid_all_u) ./ sqrt(W2_bid_all_u);

% Get correlations
[c_rand_cumsum_corr, corr_tau] = correlation(c_rand_cumsum_std);
c_1_cumsum_corr = correlation(c_1_cumsum_std);
c_2_cumsum_corr = correlation(c_2_cumsum_std);
c_1_2_cumsum_corr = correlation(c_1_2_cumsum_std);
c_bid_1_cumsum_corr = correlation(c_bid_1_cumsum_std);
c_bid_1_u_cumsum_corr = correlation(c_bid_1_u_cumsum_std);
c_bid_all_cumsum_corr = correlation(c_bid_all_cumsum_std);
c_bid_all_u_cumsum_corr = correlation(c_bid_all_u_cumsum_std);

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
    'Marker', '*',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "bid-1-always"];
p = plot(W1_bid_1_u(end), W2_bid_1_u(end),...
    'LineStyle', 'none',...
    'Marker', '*',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "bid-1-if-urgent"];
p = plot(W1_bid_all(end), W2_bid_all(end),...
    'LineStyle', 'none',...
    'Marker', '*',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "bid-all-always"];
p = plot(W1_bid_all_u(end), W2_bid_all_u(end),...
    'LineStyle', 'none',...
    'Marker', '*',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
lgd_text = [lgd_text, "bid-all-if-urgent"];
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
    'Marker', '*',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
p = plot(W1_bid_1_u_norm(end), W2_bid_1_u_norm(end),...
    'LineStyle', 'none',...
    'Marker', '*',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
p = plot(W1_bid_all_norm(end), W2_bid_all_norm(end),...
    'LineStyle', 'none',...
    'Marker', '*',...
    'MarkerSize', 10);
p.MarkerFaceColor = p.Color;
p = plot(W1_bid_all_u_norm(end), W2_bid_all_u_norm(end),...
    'LineStyle', 'none',...
    'Marker', '*',...
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
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,2);
plot(c_1_cumsum);
hold on;
plot(W1_1, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,3);
plot(c_2_cumsum);
hold on;
plot(W1_2, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-cost';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,4);
plot(c_1_2_cumsum);
hold on;
plot(W1_1_2, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency-then-cost';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,5);
plot(c_bid_1_cumsum);
hold on;
plot(W1_bid_1, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-1-always';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,6);
plot(c_bid_1_u_cumsum);
hold on;
plot(W1_bid_1_u, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-1-if-urgent';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,7);
plot(c_bid_all_cumsum);
hold on;
plot(W1_bid_all, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-all-always';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,8);
plot(c_bid_all_u_cumsum);
hold on;
plot(W1_bid_all_u, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-all-if-urgent';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Cumulative cost';
axes.YLabel.FontSize = 12;

%% Unfairness vs. time for cumulatice cost
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, default_height/2, default_width, default_height];
plot(W2_rand);
hold on;
plot(W2_1);
plot(W2_2);
plot(W2_1_2);
plot(W2_bid_1);
plot(W2_bid_1_u);
plot(W2_bid_all);
plot(W2_bid_all_u);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'Unfairness vs. time';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Unfairness';
axes.YLabel.FontSize = 12;
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
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,2);
plot(c_1_cumsum_norm);
hold on;
plot(W1_1_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,3);
plot(c_2_cumsum_norm);
hold on;
plot(W1_2_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-cost';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,4);
plot(c_1_2_cumsum_norm);
hold on;
plot(W1_1_2_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency-then-cost';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,5);
plot(c_bid_1_cumsum_norm);
hold on;
plot(W1_bid_1_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-1-always';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,6);
plot(c_bid_1_u_cumsum_norm);
hold on;
plot(W1_bid_1_u_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-1-if-urgent';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,7);
plot(c_bid_all_cumsum_norm);
hold on;
plot(W1_bid_all_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-all-always';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,8);
plot(c_bid_all_u_cumsum_norm);
hold on;
plot(W1_bid_all_u_norm, 'Linewidth', 3);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-all-if-urgent';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized cumulative cost';
axes.YLabel.FontSize = 12;

%% Normalized unfairness vs. time (for normalized cumulatice cost)
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, default_height/2, default_width, default_height];
plot(W2_rand_norm);
hold on;
plot(W2_1_norm);
plot(W2_2_norm);
plot(W2_1_2_norm);
plot(W2_bid_1_norm);
plot(W2_bid_1_u_norm);
plot(W2_bid_all_norm);
plot(W2_bid_all_u_norm);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'Normalized unfairness vs. time';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Normalized unfairness';
axes.YLabel.FontSize = 12;
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
mean_num_inter = mean(num_inter, 2);
plot(mean_num_inter, 'Linewidth', 10);
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

%% Standardized cumulative cost plot
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
subplot(4,2,1);
plot(c_rand_cumsum_std);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'baseline-random';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Standardized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,2);
plot(c_1_cumsum_std);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Standardized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,3);
plot(c_2_cumsum_std);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-cost';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Standardized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,4);
plot(c_1_2_cumsum_std);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'centralized-urgency-then-cost';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Standardized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,5);
plot(c_bid_1_cumsum_std);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-1-always';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Standardized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,6);
plot(c_bid_1_u_cumsum_std);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-1-if-urgent';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Standardized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,7);
plot(c_bid_all_cumsum_std);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-all-always';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Standardized cumulative cost';
axes.YLabel.FontSize = 12;
subplot(4,2,8);
plot(c_bid_all_u_cumsum_std);
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'bid-all-if-urgent';
axes.Title.FontSize = 16;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Time period';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Standardized cumulative cost';
axes.YLabel.FontSize = 12;

% %% Histograms of accumulated costs
% figure(fg);
% fg = fg + 1;
% fig = gcf;
% fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
% subplot(4,2,1);
% [values, counts] = count_values(c_rand_cumsum(end,:));
% stem(values, counts);
% hold on;
% stem(W1_rand(end), max(counts), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'baseline-random';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(4,2,2);
% [values, counts] = count_values(c_1_cumsum(end,:));
% stem(values, counts);
% hold on;
% stem(W1_1(end), max(counts), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'centralized-urgency';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(4,2,3);
% [values, counts] = count_values(c_2_cumsum(end,:));
% stem(values, counts);
% hold on;
% stem(W1_2(end), max(counts), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'centralized-cost';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(4,2,4);
% [values, counts] = count_values(c_1_2_cumsum(end,:));
% stem(values, counts);
% hold on;
% stem(W1_1_2(end), max(counts), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'centralized-urgency-then-cost';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(4,2,5);
% [values, counts] = count_values(c_bid_all_cumsum(end,:));
% stem(values, counts);
% hold on;
% stem(W1_bid_all(end), max(counts), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'bid-all-always';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(4,2,6);
% [values, counts] = count_values(c_bid_all_u_cumsum(end,:));
% stem(values, counts);
% hold on;
% stem(W1_bid_all_u(end), max(counts), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'bid-all-if-urgent';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(4,2,[7 8]);
% [values, counts] = count_values(num_inter(end,:));
% stem(values, counts);
% hold on;
% stem(mean_num_inter(end), max(counts), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'Number of interactions per agent';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Number of interactions';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;

% %% Histograms of normalized accumulated costs
% figure(fg);
% fg = fg + 1;
% fig = gcf;
% fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
% subplot(3,2,1);
% h = histogram(c_rand_cumsum_norm(end,:));
% hold on;
% stem(W1_rand_norm(end), max(h.Values), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'baseline-random';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Normalized cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(3,2,2);
% h = histogram(c_1_cumsum_norm(end,:));
% hold on;
% stem(W1_1_norm(end), max(h.Values), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'centralized-urgency';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Normalized cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(3,2,3);
% h = histogram(c_2_cumsum_norm(end,:));
% hold on;
% stem(W1_2_norm(end), max(h.Values), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'centralized-cost';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Normalized cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(3,2,4);
% h = histogram(c_1_2_cumsum_norm(end,:));
% hold on;
% stem(W1_1_2_norm(end), max(h.Values), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'centralized-urgency-then-cost';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Normalized cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(3,2,5);
% h = histogram(c_bid_all_cumsum_norm(end,:));
% hold on;
% stem(W1_bid_all_norm(end), max(h.Values), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'bid-all-always';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Normalized cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;
% subplot(3,2,6);
% h = histogram(c_bid_all_u_cumsum_norm(end,:));
% hold on;
% stem(W1_bid_all_u_norm(end), max(h.Values), 'Linewidth', 3, 'Marker', 'none');
% axes = gca;
% axis tight;
% axes.Title.Interpreter = 'latex';
% axes.Title.String = 'bid-all-if-urgent';
% axes.Title.FontSize = 16;
% axes.XAxis.TickLabelInterpreter = 'latex';
% axes.XAxis.FontSize = 10;
% axes.YAxis.TickLabelInterpreter = 'latex';
% axes.YAxis.FontSize = 10;
% axes.XLabel.Interpreter = 'latex';
% axes.XLabel.String = 'Normalized cumulative cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;

%% Time-correlation of accumulated costs
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, default_height/2, default_width, default_height];
plot(corr_tau, c_rand_cumsum_corr);
hold on;
plot(corr_tau, c_1_cumsum_corr);
plot(corr_tau, c_2_cumsum_corr);
plot(corr_tau, c_1_2_cumsum_corr);
plot(corr_tau, c_bid_1_cumsum_corr);
plot(corr_tau, c_bid_1_u_cumsum_corr);
plot(corr_tau, c_bid_all_cumsum_corr);
plot(corr_tau, c_bid_all_u_cumsum_corr);
stem(0, c_rand_cumsum_corr(corr_tau == 0));
axes = gca;
axis tight;
axes.Title.Interpreter = 'latex';
axes.Title.String = 'Correlation over time';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';

axes.XLabel.String = 'Time shift';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Correlation';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.Interpreter = 'latex';
lgd.FontSize = 12;
lgd.Location = 'bestoutside';

%% Inform user when done
fprintf('DONE\n\n');

%% Helper functions
% Returns all maximizers (if there are multiple)
function [max_v, max_i] = multi_maxes(input)
    [max_v, max_i] = max(input);
    input(max_i) = -realmax;
    [next_max_v, next_max_i] = max(input);
    while next_max_v == max_v
        max_i = [max_i, next_max_i];
        input(next_max_i) = -realmax;
        [next_max_v, next_max_i] = max(input);
    end
end

% Checks if there are multiple maximizers and returns one uniformly at
% random
function [max_v, max_i] = multi_max(input)
    % Get maximizers
    [max_v, max_i] = multi_maxes(input);
    % Choose one uniformly at random if there are multiple
    num_max = length(max_i);
    if num_max > 1
        max_i = max_i(ceil(rand(1) * num_max));
    end
end

% Gets delayed agents given index of passing agent
function d = get_d(I, p_i)    
    d = I;
    d(p_i) = [];
end

% Gets karma paid by passing agent to delayed agents
function [k_p, k_d] = get_karma_payments(m_p, d, curr_k, param)
    % Distribute karma evenly over delayed agents. If an agent will max
    % out their karma, tough luck!
    k_p_per_d = floor(m_p / param.num_d);
    k_d = zeros(1, param.num_d);
    for i = 1 : param.num_d
        k_d(i) = min([k_p_per_d, param.k_max - curr_k(d(i))]);
    end
    % Sum back the total karma distributed, which takes into account
    % delayed agents for which karma will saturate. This is the final
    % total paid by passing agent
    k_p = sum(k_d);
end

% Computes the correlation of a signal over time
function [corr, tau] = correlation(input)
    T = size(input, 1);
    center_t = ceil(T / 2);
    tau = -center_t + 1 : 1 : center_t;
    center_signal = input(center_t,:);
    corr = center_signal * input.';
end

% This is a 'custom' histogram-like function that lets us plot histogram as
% a stem plot
function [values, counts] = count_values(input)
    values = unique(input);
    values = sort(values);
    counts = zeros(size(values));
    for i = 1 : length(values)
        counts(i) = sum(input == values(i));
    end
end

% Sets axis limit 'scale' above/below min-max values
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

% Sets y-axis limit 'scale' above/below min-max values
function y_semi_tight(ax, scale)
    axis tight; % Set axis tight
    yl = ylim(ax); % Get tight axis limits
    range = yl(2) - yl(1); % Get tight axis range
    sc_range = scale * range; % Scale range
    yl(1) = yl(1) - (sc_range - range) / 2; % New ymin
    yl(2) = yl(1) + sc_range; % New ymax
    ylim(ax, yl);
end