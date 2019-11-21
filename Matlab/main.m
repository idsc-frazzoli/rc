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

%% Code control bits
% Autocorrelation takes long time to compute
control.compute_autocorrelation = false;

% Flag to simulate centralized limited memory policies
control.lim_mem_policies = true;

% Flag to simulate heuristic karma policies
control.karma_heuristic_policies = false;

%% Parameters
param = load_parameters();

%% Simulation initialization
% Populatin of agent indices to sample from
population = 1 : param.N;

% Cost matrices for different policies
% Row => Time step
% Col => Agent

% Centralized policies
% Cost for baseline policy - random
c_rand = zeros(param.tot_num_inter, param.N);
% Cost for centralized policy 1 - centralized urgency
c_1 = zeros(param.tot_num_inter, param.N);
% Cost for centralized policy 2 - centralized cost
c_2 = zeros(param.tot_num_inter, param.N);
% Cost for centralized policy 1_2 - centralized urgency then cost
c_1_2 = zeros(param.tot_num_inter, param.N);

% Centralized policies with limited memory
if control.lim_mem_policies
    c_lim_mem = cell(param.lim_mem_num_steps, 1);
    c_in_mem  = cell(param.lim_mem_num_steps, 1);
    for i = 1 : param.lim_mem_num_steps
        c_lim_mem{i} = zeros(param.tot_num_inter, param.N);
        c_in_mem{i} = zeros(param.lim_mem_steps(i), param.N);
    end
end

% Hueristic karma policies
if control.karma_heuristic_policies
    % Cost for bid 1 always policy
    c_bid_1 = zeros(param.tot_num_inter, param.N);
    % Cost for bid 1 if urgent policy
    c_bid_1_u = zeros(param.tot_num_inter, param.N);
    % Cost for bid all always policy
    c_bid_all = zeros(param.tot_num_inter, param.N);
    % Cost for bid all if urgent policy
    c_bid_all_u = zeros(param.tot_num_inter, param.N);
end

% Karma matrices for karma polices. Initialized uniformly randomly
% between k_min & k_max. Same initialization for all policies
karma_init = round(rand(1, param.N) * (param.k_max - param.k_min)) + param.k_min;

if control.karma_heuristic_policies
    % Karma for bid 1 always policy
    k_bid_1 = zeros(param.tot_num_inter, param.N);
    k_bid_1(1,:) = karma_init;
    % Karma for bid 1 if urgent policy
    k_bid_1_u = zeros(param.tot_num_inter, param.N);
    k_bid_1_u(1,:) = karma_init;
    % Karma for bid all always policy
    k_bid_all = zeros(param.tot_num_inter, param.N);
    k_bid_all(1,:) = karma_init;
    % Karma for bid all if urgent policy
    k_bid_all_u = zeros(param.tot_num_inter, param.N);
    k_bid_all_u(1,:) = karma_init;
end

% Number of times each agent was in an intersection, as a accumulated sum
num_inter = zeros(param.tot_num_inter, param.N);

%% Simulation run
% Convention:   p := agent that passes
%               d := agent(s) that are delayed
for day = 1 : param.num_days
    % Pick urgency in {0,U} uniformly at random for all agents. Urgency
    % stays constant for agents per day
    u_today = round(rand(1, param.N)) * param.U;
    
    for inter = 1 : param.num_inter_per_day
        t = (day - 1) * param.num_inter_per_day + inter;
        % Tell user where we are
        fprintf('Day: %d Interaction: %d Timestep: %d\n', day, inter, t);

        if ~param.same_num_inter
            % Sample agents i & j uniformly at random
            I = randsample(population, param.I_size);
        else
            % If all population has been sampled, re-fill population
            if isempty(population)
                population = 1 : param.N;
            end
            % Sample agents i & j uniformly at random and remove them from
            % population
            I = randsample(population, param.I_size);
            for i = 1 : param.I_size
                population(population == I(i)) = [];
            end
        end

        % Increment number of interactions for picked agents
        % Reset to zero (which is what num_inter is initialized to) at end
        % of warm-up period
        if t > 1 && t ~= param.t_warm_up
            num_inter(t,:) = num_inter(t-1,:);
        end
        num_inter(t,I) = num_inter(t,I) + 1;

        % Urgency of sampled agents
        u = u_today(I);

        %% Random policy
        % Choose an agent to pass uniformly at random
        p = I(ceil(rand(1) * length(I)));

        % Agents incur cost equal to their urgency, except passing agent
        c_rand(t,I) = u;
        c_rand(t,p) = 0;

        %% CENTRALIZED POLICIES %%
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
        if t <= param.t_warm_up
            a_u = sum(c_2(1:t,I)) + u;
        else
            a_u = sum(c_2(param.t_warm_up+1:t,I)) + u;
        end
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
            if t <= param.t_warm_up
                a_u = sum(c_1_2(1:t,p_max_u)) + u(p_ind);
            else
                a_u = sum(c_1_2(param.t_warm_up+1:t,p_max_u)) + u(p_ind);
            end
            [~, p_i] = multi_max(a_u);
            p = p_max_u(p_i);
        else
            p = p_max_u;
        end

        % Agents incur cost equal to their urgency, except passing agent
        c_1_2(t,I) = u;
        c_1_2(t,p) = 0;

        %% Centralized policies with limited memroy
        if control.lim_mem_policies
            % Minimize accumulated cost up to limited number of interactions per
            % agent, coin-flip if tie
            for i = 1 : param.lim_mem_num_steps
                % Agent with maximum accumulated cost in memory (counting current
                % urgency) passes
                a_u = sum(c_in_mem{i}(:,I)) + u;
                [~, p_i] = multi_max(a_u);
                p = I(p_i);

                % Agents incur cost equal to their urgency, except passing agent
                c_lim_mem{i}(t,I) = u;
                c_lim_mem{i}(t,p) = 0;

                % Update limited memory with most recent cost
                c_in_mem{i}(1:end-1,I) = c_in_mem{i}(2:end,I);
                c_in_mem{i}(end,I) = c_lim_mem{i}(t,I);
            end
        end

        %% HEURISTIC KARMA POLICIES
        if control.karma_heuristic_policies
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
            if t < param.tot_num_inter
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
            if t < param.tot_num_inter
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
            if t < param.tot_num_inter
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
            if t < param.tot_num_inter
                [k_p, k_d] = get_karma_payments(m_p, d, k_bid_all_u(t,:), param);
                k_bid_all_u(t+1,:) = k_bid_all_u(t,:);
                k_bid_all_u(t+1,p) = k_bid_all_u(t+1,p) - k_p;
                k_bid_all_u(t+1,d) = k_bid_all_u(t+1,d) + k_d;
            end
        end
    end
end

%% Perfromance measures
% Accumulated costs per agent at each time step
a_rand = get_accumulated_cost(c_rand, param);
a_1 = get_accumulated_cost(c_1, param);
a_2 = get_accumulated_cost(c_2, param);
a_1_2 = get_accumulated_cost(c_1_2, param);
if control.lim_mem_policies
    a_lim_mem = cell(param.lim_mem_num_steps, 1);
    for i = 1 : param.lim_mem_num_steps
        a_lim_mem{i} = get_accumulated_cost(c_lim_mem{i}, param);
    end
end
if control.karma_heuristic_policies
    a_bid_1 = get_accumulated_cost(c_bid_1, param);
    a_bid_1_u = get_accumulated_cost(c_bid_1_u, param);
    a_bid_all = get_accumulated_cost(c_bid_all, param);
    a_bid_all_u = get_accumulated_cost(c_bid_all_u, param);
end

% If number of interactions per agent is fixed, true time is interprated as
% the time after which all agents have participated in an interaction
if param.same_num_inter
    actual_t = param.num_inter_in_N : param.num_inter_in_N : param.tot_num_inter;
    num_inter = num_inter(actual_t,:);
    a_rand = a_rand(actual_t,:);
    a_1 = a_1(actual_t,:);
    a_2 = a_2(actual_t,:);
    a_1_2 = a_1_2(actual_t,:);
    if control.lim_mem_policies
        for i = 1 : param.lim_mem_num_steps
            a_lim_mem{i} = a_lim_mem{i}(actual_t,:);
        end
    end
    if control.karma_heuristic_policies
        a_bid_1 = a_bid_1(actual_t,:);
        a_bid_1_u = a_bid_1_u(actual_t,:);
        a_bid_all = a_bid_all(actual_t,:);
        a_bid_all_u = a_bid_all_u(actual_t,:);
    end
end

% Accumulated costs per agent at each time step, normalized by their
% respective number of interactions
% Note that nan can result if agent(s) were never in an intersection, which
% is handled later by using 'nan---' functions (which ignore nan's)
a_rand_norm = a_rand ./ num_inter;
a_1_norm = a_1 ./ num_inter;
a_2_norm = a_2 ./ num_inter;
a_1_2_norm = a_1_2 ./ num_inter;
if control.lim_mem_policies
    a_lim_mem_norm = cell(param.lim_mem_num_steps, 1);
    for i = 1 : param.lim_mem_num_steps
        a_lim_mem_norm{i} = a_lim_mem{i} ./ num_inter;
    end
end
if control.karma_heuristic_policies
    a_bid_1_norm = a_bid_1 ./ num_inter;
    a_bid_1_u_norm = a_bid_1_u ./ num_inter;
    a_bid_all_norm = a_bid_all ./ num_inter;
    a_bid_all_u_norm = a_bid_all_u ./ num_inter;
end

% Inefficiency vs. time
W1_rand = nanmean(a_rand, 2);
W1_1 = nanmean(a_1, 2);
W1_2 = nanmean(a_2, 2);
W1_1_2 = nanmean(a_1_2, 2);
if control.lim_mem_policies
    W1_lim_mem = cell(param.lim_mem_num_steps, 1);
    for i = 1 : param.lim_mem_num_steps
        W1_lim_mem{i} = nanmean(a_lim_mem{i}, 2);
    end
end
if control.karma_heuristic_policies
    W1_bid_1 = nanmean(a_bid_1, 2);
    W1_bid_1_u = nanmean(a_bid_1_u, 2);
    W1_bid_all = nanmean(a_bid_all, 2);
    W1_bid_all_u = nanmean(a_bid_all_u, 2);
end

% Normalized inefficiency vs. time
W1_rand_norm = nanmean(a_rand_norm, 2);
W1_1_norm = nanmean(a_1_norm, 2);
W1_2_norm = nanmean(a_2_norm, 2);
W1_1_2_norm = nanmean(a_1_2_norm, 2);
if control.lim_mem_policies
    W1_lim_mem_norm = cell(param.lim_mem_num_steps, 1);
    for i = 1 : param.lim_mem_num_steps
        W1_lim_mem_norm{i} = nanmean(a_lim_mem_norm{i}, 2);
    end
end
if control.karma_heuristic_policies
    W1_bid_1_norm = nanmean(a_bid_1_norm, 2);
    W1_bid_1_u_norm = nanmean(a_bid_1_u_norm, 2);
    W1_bid_all_norm = nanmean(a_bid_all_norm, 2);
    W1_bid_all_u_norm = nanmean(a_bid_all_u_norm, 2);
end

% Unfairness vs. time
W2_rand = nanvar(a_rand, [], 2);
W2_1 = nanvar(a_1, [], 2);
W2_2 = nanvar(a_2, [], 2);
W2_1_2 = nanvar(a_1_2, [], 2);
if control.lim_mem_policies
    W2_lim_mem = cell(param.lim_mem_num_steps, 1);
    for i = 1 : param.lim_mem_num_steps
        W2_lim_mem{i} = nanvar(a_lim_mem{i}, [], 2);
    end
end
if control.karma_heuristic_policies
    W2_bid_1 = nanvar(a_bid_1, [], 2);
    W2_bid_1_u = nanvar(a_bid_1_u, [], 2);
    W2_bid_all = nanvar(a_bid_all, [], 2);
    W2_bid_all_u = nanvar(a_bid_all_u, [], 2);
end

% Normalized unfairness vs. time
W2_rand_norm = nanvar(a_rand_norm, [], 2);
W2_1_norm = nanvar(a_1_norm, [], 2);
W2_2_norm = nanvar(a_2_norm, [], 2);
W2_1_2_norm = nanvar(a_1_2_norm, [], 2);
if control.lim_mem_policies
    W2_lim_mem_norm = cell(param.lim_mem_num_steps, 1);
    for i = 1 : param.lim_mem_num_steps
        W2_lim_mem_norm{i} = nanvar(a_lim_mem_norm{i}, [], 2);
    end
end
if control.karma_heuristic_policies
    W2_bid_1_norm = nanvar(a_bid_1_norm, [], 2);
    W2_bid_1_u_norm = nanvar(a_bid_1_u_norm, [], 2);
    W2_bid_all_norm = nanvar(a_bid_all_norm, [], 2);
    W2_bid_all_u_norm = nanvar(a_bid_all_u_norm, [], 2);
end

% Standardized accumulated costs, i.e. accumultated costs brought to zero
% mean and unit std-dev. Required for autocorrelation and allows to
% investigate 'mixing' of policies
a_rand_std = standardize(a_rand, W1_rand, W2_rand);
a_1_std = standardize(a_1, W1_1, W2_1);
a_2_std = standardize(a_2, W1_2, W2_2);
a_1_2_std = standardize(a_1_2, W1_1_2, W2_1_2);
if control.lim_mem_policies
    a_lim_mem_std = cell(param.lim_mem_num_steps, 1);
    for i = 1 : param.lim_mem_num_steps
        a_lim_mem_std{i} = standardize(a_lim_mem{i}, W1_lim_mem{i}, W2_lim_mem{i});
    end
end
if control.karma_heuristic_policies
    a_bid_1_std = standardize(a_bid_1, W1_bid_1, W2_bid_1);
    a_bid_1_u_std = standardize(a_bid_1_u, W1_bid_1_u, W2_bid_1_u);
    a_bid_all_std = standardize(a_bid_all, W1_bid_all, W2_bid_all);
    a_bid_all_u_std = standardize(a_bid_all_u, W1_bid_all_u, W2_bid_all_u);
end

%% Autocorrelation of accumulated cost
% Provides indication on how well population cost 'mixes' with time
if control.compute_autocorrelation
    fprintf('Computing autocorrelation for baseline-random\n');
    [c_rand_cumsum_acorr, acorr_tau] = autocorrelation(a_rand_std);
    fprintf('Computing autocorrelation for centralized-urgency\n');
    c_1_cumsum_acorr = autocorrelation(a_1_std);
    fprintf('Computing autocorrelation for centralized-cost\n');
    c_2_cumsum_acorr = autocorrelation(a_2_std);
    fprintf('Computing autocorrelation for centralized-urgency-then-cost\n');
    c_1_2_cumsum_acorr = autocorrelation(a_1_2_std);
    if control.lim_mem_policies
        c_lim_mem_cumsum_acorr = cell(param.lim_mem_num_steps, 1);
        for i = 1 : param.lim_mem_num_steps
            fprintf('Computing autocorrelation for centralized-cost-mem-%d\n', param.lim_mem_steps(i));
            c_lim_mem_cumsum_acorr{i} = autocorrelation(a_lim_mem_std{i});
        end
    end
    if control.karma_heuristic_policies
        fprintf('Computing autocorrelation for bid-1-always\n');
        c_bid_1_cumsum_acorr = autocorrelation(a_bid_1_std);
        fprintf('Computing autocorrelation for bid-1-if-urgent\n');
        c_bid_1_u_cumsum_acorr = autocorrelation(a_bid_1_u_std);
        fprintf('Computing autocorrelation for bid-all-always\n');
        c_bid_all_cumsum_acorr = autocorrelation(a_bid_all_std);
        fprintf('Computing autocorrelation for bid-all-if-urgent\n');
        c_bid_all_u_cumsum_acorr = autocorrelation(a_bid_all_u_std);
    end
end

%% Scatter plot - Inefficiency vs unfairness (non-normalized)
fprintf('Plotting\n');
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
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
if control.lim_mem_policies
    for i = 1 : param.lim_mem_num_steps
        p = plot(W1_lim_mem{i}(end), W2_lim_mem{i}(end),...
            'LineStyle', 'none',...
            'Marker', '*',...
            'MarkerSize', 10);
        p.MarkerFaceColor = p.Color;
        lgd_text = [lgd_text, strcat("centralized-cost-mem-", int2str(param.lim_mem_steps(i)))];
    end
end
if control.karma_heuristic_policies
    p = plot(W1_bid_1(end), W2_bid_1(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    p.MarkerFaceColor = p.Color;
    lgd_text = [lgd_text, "bid-1-always"];
    p = plot(W1_bid_1_u(end), W2_bid_1_u(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    p.MarkerFaceColor = p.Color;
    lgd_text = [lgd_text, "bid-1-if-urgent"];
    p = plot(W1_bid_all(end), W2_bid_all(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    p.MarkerFaceColor = p.Color;
    lgd_text = [lgd_text, "bid-all-always"];
    p = plot(W1_bid_all_u(end), W2_bid_all_u(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    p.MarkerFaceColor = p.Color;
    lgd_text = [lgd_text, "bid-all-if-urgent"];
end
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

%% Scatter plot - Normalized inefficiency vs unfairness
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
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
if control.lim_mem_policies
    for i = 1 : param.lim_mem_num_steps
        p = plot(W1_lim_mem_norm{i}(end), W2_lim_mem_norm{i}(end),...
            'LineStyle', 'none',...
            'Marker', '*',...
            'MarkerSize', 10);
        p.MarkerFaceColor = p.Color;
    end
end
if control.karma_heuristic_policies
    p = plot(W1_bid_1_norm(end), W2_bid_1_norm(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    p.MarkerFaceColor = p.Color;
    p = plot(W1_bid_1_u_norm(end), W2_bid_1_u_norm(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    p.MarkerFaceColor = p.Color;
    p = plot(W1_bid_all_norm(end), W2_bid_all_norm(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    p.MarkerFaceColor = p.Color;
    p = plot(W1_bid_all_u_norm(end), W2_bid_all_u_norm(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    p.MarkerFaceColor = p.Color;
end
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

%% Accumulated cost plot - Gives indication on how fast variance grows
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
subplot(2,2,1);
plot(a_rand);
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
axes.YLabel.String = 'Accumulated cost';
axes.YLabel.FontSize = 12;
subplot(2,2,2);
plot(a_1);
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
axes.YLabel.String = 'Accumulated cost';
axes.YLabel.FontSize = 12;
subplot(2,2,3);
plot(a_2);
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
axes.YLabel.String = 'Accumulated cost';
axes.YLabel.FontSize = 12;
subplot(2,2,4);
plot(a_1_2);
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
axes.YLabel.String = 'Accumulated cost';
axes.YLabel.FontSize = 12;

%% Accumulated cost plot for limited memory policies
if control.lim_mem_policies
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [0, 0, screenwidth, screenheight];
    num_cols = round(sqrt(screenwidth / screenheight * param.lim_mem_num_steps));
    num_rows = ceil(param.lim_mem_num_steps / num_cols);
    for i = 1 : param.lim_mem_num_steps
        subplot(num_rows,num_cols,i);
        plot(a_lim_mem{i});
        hold on;
        plot(W1_lim_mem{i}, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = ['centralized-cost-mem-', int2str(param.lim_mem_steps(i))];
        axes.Title.FontSize = 16;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time period';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Accumulated cost';
        axes.YLabel.FontSize = 12;
    end
end

%% Accumulated cost plot for heuristic karma policies
if control.karma_heuristic_policies
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
    subplot(2,2,1);
    plot(a_bid_1);
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
    axes.YLabel.String = 'Accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,2);
    plot(a_bid_1_u);
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
    axes.YLabel.String = 'Accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,3);
    plot(a_bid_all);
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
    axes.YLabel.String = 'Accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,4);
    plot(a_bid_all_u);
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
    axes.YLabel.String = 'Accumulated cost';
    axes.YLabel.FontSize = 12;
end

%% Unfairness vs. time for accumulated cost
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
plot(W2_rand, 'LineWidth', 2);
hold on;
plot(W2_1, 'LineWidth', 2);
plot(W2_2, 'LineWidth', 2);
plot(W2_1_2, 'LineWidth', 2);
if control.lim_mem_policies
    for i = 1 : param.lim_mem_num_steps
        plot(W2_lim_mem{i}, '--');
    end
end
if control.karma_heuristic_policies
    plot(W2_bid_1, '-.');
    plot(W2_bid_1_u, '-.');
    plot(W2_bid_all, '-.');
    plot(W2_bid_all_u, '-.');
end
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

%% Normalized accumulated cost plot
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
subplot(2,2,1);
plot(a_rand_norm);
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
axes.YLabel.String = 'Normalized accumulated cost';
axes.YLabel.FontSize = 12;
subplot(2,2,2);
plot(a_1_norm);
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
axes.YLabel.String = 'Normalized accumulated cost';
axes.YLabel.FontSize = 12;
subplot(2,2,3);
plot(a_2_norm);
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
axes.YLabel.String = 'Normalized accumulated cost';
axes.YLabel.FontSize = 12;
subplot(2,2,4);
plot(a_1_2_norm);
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
axes.YLabel.String = 'Normalized accumulated cost';
axes.YLabel.FontSize = 12;

%% Normalized accumulated cost plot for limited memory policies
if control.lim_mem_policies
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [0, 0, screenwidth, screenheight];
    for i = 1 : param.lim_mem_num_steps
        subplot(num_rows,num_cols,i);
        plot(a_lim_mem_norm{i});
        hold on;
        plot(W1_lim_mem_norm{i}, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = ['centralized-cost-mem-', int2str(param.lim_mem_steps(i))];
        axes.Title.FontSize = 16;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time period';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Normalized accumulated cost';
        axes.YLabel.FontSize = 12;
    end
end

%% Normalized accumulated cost plot for heuristic karma policies
if control.karma_heuristic_policies
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
    subplot(2,2,1);
    plot(a_bid_1_norm);
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
    axes.YLabel.String = 'Normalized accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,2);
    plot(a_bid_1_u_norm);
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
    axes.YLabel.String = 'Normalized accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,3);
    plot(a_bid_all_norm);
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
    axes.YLabel.String = 'Normalized accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,4);
    plot(a_bid_all_u_norm);
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
    axes.YLabel.String = 'Normalized accumulated cost';
    axes.YLabel.FontSize = 12;
end

%% Normalized unfairness vs. time (for normalized accumulated cost)
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
plot(W2_rand_norm, 'LineWidth', 2);
hold on;
plot(W2_1_norm, 'LineWidth', 2);
plot(W2_2_norm, 'LineWidth', 2);
plot(W2_1_2_norm, 'LineWidth', 2);
if control.lim_mem_policies
    for i = 1 : param.lim_mem_num_steps
        plot(W2_lim_mem_norm{i}, '--');
    end
end
if control.karma_heuristic_policies
    plot(W2_bid_1_norm, '-.');
    plot(W2_bid_1_u_norm, '-.');
    plot(W2_bid_all_norm, '-.');
    plot(W2_bid_all_u_norm, '-.');
end
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
plot(mean_num_inter, 'Linewidth', 3);
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

%% Standardized accumulated cost plot
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
subplot(2,2,1);
plot(a_rand_std);
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
axes.YLabel.String = 'Standardized accumulated cost';
axes.YLabel.FontSize = 12;
subplot(2,2,2);
plot(a_1_std);
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
axes.YLabel.String = 'Standardized accumulated cost';
axes.YLabel.FontSize = 12;
subplot(2,2,3);
plot(a_2_std);
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
axes.YLabel.String = 'Standardized accumulated cost';
axes.YLabel.FontSize = 12;
subplot(2,2,4);
plot(a_1_2_std);
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
axes.YLabel.String = 'Standardized accumulated cost';
axes.YLabel.FontSize = 12;

%% Standardized accumulated cost plot for limited memory policies
if control.lim_mem_policies
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [0, 0, screenwidth, screenheight];
    for i = 1 : param.lim_mem_num_steps
        subplot(num_rows,num_cols,i);
        plot(a_lim_mem_std{i});
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = ['centralized-cost-mem-', int2str(param.lim_mem_steps(i))];
        axes.Title.FontSize = 16;
        axes.XAxis.TickLabelInterpreter = 'latex';
        axes.XAxis.FontSize = 10;
        axes.YAxis.TickLabelInterpreter = 'latex';
        axes.YAxis.FontSize = 10;
        axes.XLabel.Interpreter = 'latex';
        axes.XLabel.String = 'Time period';
        axes.XLabel.FontSize = 12;
        axes.YLabel.Interpreter = 'latex';
        axes.YLabel.String = 'Standardized accumulated cost';
        axes.YLabel.FontSize = 12;
    end
end

%% Standardized accumulated cost plot for heuristic karma policies
if control.karma_heuristic_policies
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [mod(fg,2)*default_width, 0, default_width, screenheight];
    subplot(2,2,1);
    plot(a_bid_1_std);
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
    axes.YLabel.String = 'Standardized accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,2);
    plot(a_bid_1_u_std);
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
    axes.YLabel.String = 'Standardized accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,3);
    plot(a_bid_all_std);
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
    axes.YLabel.String = 'Standardized accumulated cost';
    axes.YLabel.FontSize = 12;
    subplot(2,2,4);
    plot(a_bid_all_u_std);
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
    axes.YLabel.String = 'Standardized accumulated cost';
    axes.YLabel.FontSize = 12;
end

%% Autocorrelation of accumulated costs
if control.compute_autocorrelation
    figure(fg);
    fg = fg + 1;
    fig = gcf;
    fig.Position = [0, 0, screenwidth, screenheight];
    plot(acorr_tau, c_rand_cumsum_acorr, 'LineWidth', 2);
    hold on;
    plot(acorr_tau, c_1_cumsum_acorr, 'LineWidth', 2);
    plot(acorr_tau, c_2_cumsum_acorr, 'LineWidth', 2);
    plot(acorr_tau, c_1_2_cumsum_acorr, 'LineWidth', 2);
    if control.lim_mem_policies
        for i = 1 : param.lim_mem_num_steps
            plot(acorr_tau, c_lim_mem_cumsum_acorr{i}, '--');
        end
    end
    if control.karma_heuristic_policies
        plot(acorr_tau, c_bid_1_cumsum_acorr, '-.');
        plot(acorr_tau, c_bid_1_u_cumsum_acorr, '-.');
        plot(acorr_tau, c_bid_all_cumsum_acorr, '-.');
        plot(acorr_tau, c_bid_all_u_cumsum_acorr, '-.');
    end
    stem(0, c_rand_cumsum_acorr(acorr_tau == 0));
    axes = gca;
    axis tight;
    axes.Title.Interpreter = 'latex';
    axes.Title.String = 'Autocorrelation of accumulated costs';
    axes.Title.FontSize = 18;
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
% axes.XLabel.String = 'Accumulated cost';
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
% axes.XLabel.String = 'Accumulated cost';
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
% axes.XLabel.String = 'Accumulated cost';
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
% axes.XLabel.String = 'Accumulated cost';
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
% axes.XLabel.String = 'Accumulated cost';
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
% axes.XLabel.String = 'Accumulated cost';
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
% axes.XLabel.String = 'Normalized accumulated cost';
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
% axes.XLabel.String = 'Normalized accumulated cost';
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
% axes.XLabel.String = 'Normalized accumulated cost';
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
% axes.XLabel.String = 'Normalized accumulated cost';
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
% axes.XLabel.String = 'Normalized accumulated cost';
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
% axes.XLabel.String = 'Normalized accumulated cost';
% axes.XLabel.FontSize = 12;
% axes.YLabel.Interpreter = 'latex';
% axes.YLabel.String = 'Frequency';
% axes.YLabel.FontSize = 12;

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

% Gets accumulated costs counting in warm-up period reset
function a = get_accumulated_cost(c, param)
    a = [cumsum(c(1:param.t_warm_up,:)); cumsum(c(param.t_warm_up+1:end,:))];
end

% Standardizes input distribution given mean and variance vectors
function output = standardize(input, input_mean, input_var)
    output = (input - input_mean) ./ sqrt(input_var);
    output(isnan(output)) = 0;
end

% Computes the autocorrelation of signal
function [acorr, tau] = autocorrelation(input)
    T = size(input, 1);
    N = size(input, 2);
    center_t = ceil(T / 2);
    tau = -center_t + 1 : 1 : center_t;
    mult_mat = input * input.';
    acorr = zeros(1, T);
    % Salvage symmetry of autocorrelation about zero and calculate it for
    % positive time shifts only
    tau_0_i = find(tau == 0);
    for i = tau_0_i : T
        acorr(i) = sum(diag(mult_mat, tau(i)));
    end
    for i = 1 : tau_0_i - 1
        acorr(i) = acorr(end-i);
    end
    acorr = acorr ./ ((T - abs(tau)) * N);
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