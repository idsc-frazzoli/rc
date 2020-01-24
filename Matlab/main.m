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
control.compute_autocorrelation = true;

% Flag to simulate centralized limited memory policies
control.lim_mem_policies = true;

% Flag to simulate heuristic karma policies
control.karma_heuristic_policies = true;

% Flag to simulate karma Nash equilibrium policies
control.karma_ne_policies = true;

% Flag to simulate karma social welfare policy
control.karma_sw_policy = true;

%% Parameters
param = load_parameters();

%% Simulation initialization
% Populatin of agent indices to sample from
population = 1 : param.N;

%% Cost matrices for different policies
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
    c_lim_mem = cell(param.num_lim_mem_steps, 1);
    c_in_mem  = cell(param.num_lim_mem_steps, 1);
    c_lim_mem_u = cell(param.num_lim_mem_steps, 1);
    c_in_mem_u  = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        c_lim_mem{i_lim_mem} = zeros(param.tot_num_inter, param.N);
        c_in_mem{i_lim_mem} = zeros(param.lim_mem_steps(i_lim_mem), param.N);
        c_lim_mem_u{i_lim_mem} = zeros(param.tot_num_inter, param.N);
        c_in_mem_u{i_lim_mem} = zeros(param.lim_mem_steps(i_lim_mem), param.N);
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

% Nash equilibrium karma policies
if control.karma_ne_policies
    c_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        c_ne{i_alpha} = zeros(param.tot_num_inter, param.N);
    end
end

% Social welfare karma policy
if control.karma_sw_policy
    c_sw = zeros(param.tot_num_inter, param.N);
end

%% Karma matrices for different karma policies
% Initialized with initial karma distribution

% Heuristic karma policies
if control.karma_heuristic_policies
    % Get initial karma distribution for respective k_ave from SW algorithm
    load(['karma_nash_equilibrium/results/sw_k_max_', num2str(param.k_max, '%02d'), '/k_ave_', num2str(param.k_ave, '%02d'), '.mat'], 'd_k_0');
    karma_init = datasample(param.K, param.N, 'Weights', d_k_0).';
    
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

% Nash equilibrium karma policies
if control.karma_ne_policies
    k_ne = cell(param.num_alpha, 1);
    file_str = ['karma_nash_equilibrium/results/k_max_', num2str(param.k_max, '%02d'), '_k_ave_', num2str(param.k_ave, '%02d'), '/alpha_'];
    for i_alpha = 1 : param.num_alpha
        k_ne{i_alpha} = zeros(param.tot_num_inter, param.N);
        
        % Initialize karma as per stationary distribution predicted by NE
        % algorithm
        load([file_str, num2str(param.alpha(i_alpha), '%.2f'), '.mat'], 'ne_d_up_u_k');
        k_ne{i_alpha}(1,:) = datasample(param.K, param.N, 'Weights', sum(ne_d_up_u_k)).';
    end
end

% Social welfare karma policy
if control.karma_sw_policy
    % Initialize karma as per stationary distribution predicted by SW
    % algorithm
    load(['karma_nash_equilibrium/results/sw_k_max_', num2str(param.k_max, '%02d'), '/k_ave_', num2str(param.k_ave, '%02d'), '.mat'], 'sw_d_up_u_k');
    k_sw = zeros(param.tot_num_inter, param.N);
    k_sw(1,:) = datasample(param.K, param.N, 'Weights', sum(sw_d_up_u_k)).';
end

%% Policy matrices for different karma policies
% Nash equilibrium karma policies
if control.karma_ne_policies
    pi_ne = cell(param.num_alpha, 1);
    file_str = ['karma_nash_equilibrium/results/k_max_', num2str(param.k_max, '%02d'), '_k_ave_', num2str(param.k_ave, '%02d'), '/alpha_'];
    for i_alpha = 1 : param.num_alpha
        load([file_str, num2str(param.alpha(i_alpha), '%.2f'), '.mat'], 'ne_pi_down_u_k_up_m');
        pi_ne{i_alpha} = ne_pi_down_u_k_up_m;
    end
end

% Social welfare karma policy
if control.karma_sw_policy
    load(['karma_nash_equilibrium/results/sw_k_max_', num2str(param.k_max, '%02d'), '/k_ave_', num2str(param.k_ave, '%02d'), '.mat'], 'sw_pi_down_u_k_up_m');
    pi_sw = sw_pi_down_u_k_up_m;
end

%% Number of times each agent was in an intersection, as an accumulated sum
num_inter = zeros(param.tot_num_inter, param.N);

%% Simulation run
% Convention:   win := agents that win
%               lose := agent(s) that lose
for day = 1 : param.num_days
    % Pick urgency in {0,U} uniformly at random for all agents. Urgency
    % stays constant for agents per day
    u_today = datasample(param.U, param.N).';
    
    for inter = 1 : param.num_inter_per_day
        t = (day - 1) * param.num_inter_per_day + inter;
        % Tell user where we are
        fprintf('Day: %d Interaction: %d Timestep: %d\n', day, inter, t);

        if ~param.same_num_inter
            % Sample agents i & j uniformly at random
            I = datasample(population, param.I_size, 'Replace', false);
        else
            % If all population has been sampled, re-fill population
            if isempty(population)
                population = 1 : param.N;
            end
            % Sample agents i & j uniformly at random and remove them from
            % population
            I = datasample(population, param.I_size, 'Replace', false);
            for i_agent = 1 : param.I_size
                population(population == I(i_agent)) = [];
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
        win = I(ceil(rand(1) * length(I)));

        % Agents incur cost equal to their urgency, except passing agent
        c_rand(t,I) = u;
        c_rand(t,win) = 0;

        %% CENTRALIZED POLICIES %%
        %% Centralized policy 1 - minimize W1, coin-flip if tie
        % Find agent(s) with max urgency, which are candidates for passing
        [~, i_win] = func.multi_maxes(u);
        win_max_u = I(i_win);

        % Now choose an agent uniformly at random if there are multiple.
        num_max_u = length(win_max_u);
        if num_max_u > 1
            win = win_max_u(ceil(rand(1) * num_max_u));
        else
            win = win_max_u;
        end

        % Agents incur cost equal to their urgency, except passing agent
        c_1(t,I) = u;
        c_1(t,win) = 0;

        %% Centralized policy 2 - minimize W2, coin-flip if tie
        % Agent with maximum accumulated cost (counting current urgency) passes
        if t <= param.t_warm_up
            a_u = sum(c_2(1:t,I)) + u;
        else
            a_u = sum(c_2(param.t_warm_up+1:t,I)) + u;
        end
        if param.normalize_cost
            a_u = a_u ./ num_inter(t,I);
        end
        [~, i_win] = func.multi_max(a_u);
        win = I(i_win);

        % Agents incur cost equal to their urgency, except passing agent
        c_2(t,I) = u;
        c_2(t,win) = 0;

        %% Centralized policy 1_2 - minimize W1, choose W2 minimizer on tie
        % Agent(s) with max urgency, which are candidates for passing, were
        % already found in first step of centralized policy 1
        % If there are multiple agents with max urgency, pick on based on
        % accumulated cost like in centralized policy 2
        if num_max_u > 1
            win_ind = zeros(1, num_max_u);
            for i_max_u = 1 : num_max_u
                win_ind(i_max_u) = find(I == win_max_u(i_max_u));
            end
            if t <= param.t_warm_up
                a_u = sum(c_1_2(1:t,win_max_u)) + u(win_ind);
            else
                a_u = sum(c_1_2(param.t_warm_up+1:t,win_max_u)) + u(win_ind);
            end
            if param.normalize_cost
                a_u = a_u ./ num_inter(t,win_max_u);
            end
            [~, i_win] = func.multi_max(a_u);
            win = win_max_u(i_win);
        else
            win = win_max_u;
        end

        % Agents incur cost equal to their urgency, except passing agent
        c_1_2(t,I) = u;
        c_1_2(t,win) = 0;

        %% Centralized policies with limited memroy
        if control.lim_mem_policies
            for i_lim_mem = 1 : param.num_lim_mem_steps
                %% Centralized cost with limited memory
                % Minimize accumulated cost up to limited number of interactions per
                % agent, coin-flip if tie
                % Agent with maximum accumulated cost in memory (counting current
                % urgency) passes
                a_u = sum(c_in_mem{i_lim_mem}(:,I)) + u;
                if param.normalize_cost
                    a_u = a_u ./ min([num_inter(t,I); param.lim_mem_steps(i_lim_mem) * ones(1, param.I_size)]);
                end
                [~, i_win] = func.multi_max(a_u);
                win = I(i_win);

                % Agents incur cost equal to their urgency, except passing agent
                c_lim_mem{i_lim_mem}(t,I) = u;
                c_lim_mem{i_lim_mem}(t,win) = 0;

                % Update limited memory with most recent cost
                c_in_mem{i_lim_mem}(1:end-1,I) = c_in_mem{i_lim_mem}(2:end,I);
                c_in_mem{i_lim_mem}(end,I) = c_lim_mem{i_lim_mem}(t,I);
                
                %% Centralized urgency then cost with limited memory
                % Agent(s) with max urgency, which are candidates for passing, were
                % already found in first step of centralized policy 1
                % If there are multiple agents with max urgency, pick on based on
                % accumulated cost up to limited number of interactions per
                % agent
                if num_max_u > 1
                    win_ind = zeros(1, num_max_u);
                    for i_max_u = 1 : num_max_u
                        win_ind(i_max_u) = find(I == win_max_u(i_max_u));
                    end
                    a_u = sum(c_in_mem_u{i_lim_mem}(:,win_max_u)) + u(win_ind);
                    if param.normalize_cost
                        a_u = a_u ./ min([num_inter(t,I); param.lim_mem_steps(i_lim_mem) * ones(1, param.I_size)]);
                    end
                    [~, i_win] = func.multi_max(a_u);
                    win = win_max_u(i_win);
                else
                    win = win_max_u;
                end

                % Agents incur cost equal to their urgency, except passing agent
                c_lim_mem_u{i_lim_mem}(t,I) = u;
                c_lim_mem_u{i_lim_mem}(t,win) = 0;

                % Update limited memory with most recent cost
                c_in_mem_u{i_lim_mem}(1:end-1,I) = c_in_mem_u{i_lim_mem}(2:end,I);
                c_in_mem_u{i_lim_mem}(end,I) = c_lim_mem_u{i_lim_mem}(t,I);
            end
        end

        %% HEURISTIC KARMA POLICIES
        if control.karma_heuristic_policies
            %% Bid 1 always policy
            % Agents simply bid 1, if they have it
            m = min([ones(1, param.I_size); k_bid_1(t,I) - param.k_min]);

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = func.multi_max(m);
            win = I(i_win);

            % Agents incur cost equal to their urgency, except passing agent
            c_bid_1(t,I) = u;
            c_bid_1(t,win) = 0;

            % Get delayed agents. They will be getting karma
            lose = func.get_lose(I, i_win);

            % Update karma
            if t < param.tot_num_inter
                [k_win, k_lose] = func.get_karma_payments(m_win, lose, k_bid_1(t,:), param);
                k_bid_1(t+1,:) = k_bid_1(t,:);
                k_bid_1(t+1,win) = k_bid_1(t+1,win) - k_win;
                k_bid_1(t+1,lose) = k_bid_1(t+1,lose) + k_lose;
            end

            %% Bid 1 if urgent policy
            % Agents bid 1, if they have it and they are urgent
            m = min([ones(1, param.I_size); k_bid_1_u(t,I) - param.k_min]);
            m(u == 0) = 0;

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = func.multi_max(m);
            win = I(i_win);

            % Agents incur cost equal to their urgency, except passing agent
            c_bid_1_u(t,I) = u;
            c_bid_1_u(t,win) = 0;

            % Get delayed agents. They will be getting karma
            lose = func.get_lose(I, i_win);

            % Update karma
            if t < param.tot_num_inter
                [k_win, k_lose] = func.get_karma_payments(m_win, lose, k_bid_1_u(t,:), param);
                k_bid_1_u(t+1,:) = k_bid_1_u(t,:);
                k_bid_1_u(t+1,win) = k_bid_1_u(t+1,win) - k_win;
                k_bid_1_u(t+1,lose) = k_bid_1_u(t+1,lose) + k_lose;
            end

            %% Bid all always policy
            % Agents simply bid all their karma, less the minimum allowed level (if
            % applicable)
            m = k_bid_all(t,I) - param.k_min;

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = func.multi_max(m);
            win = I(i_win);

            % Agents incur cost equal to their urgency, except passing agent
            c_bid_all(t,I) = u;
            c_bid_all(t,win) = 0;

            % Get delayed agents. They will be getting karma
            lose = func.get_lose(I, i_win);

            % Update karma
            if t < param.tot_num_inter
                [k_win, k_lose] = func.get_karma_payments(m_win, lose, k_bid_all(t,:), param);
                k_bid_all(t+1,:) = k_bid_all(t,:);
                k_bid_all(t+1,win) = k_bid_all(t+1,win) - k_win;
                k_bid_all(t+1,lose) = k_bid_all(t+1,lose) + k_lose;
            end

            %% Bid all if urgent policy
            % Agents bid all their karma, less the minimum allowed level (if
            % applicable), if they are urgent
            m = k_bid_all_u(t,I) - param.k_min;
            m(u == 0) = 0;

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = func.multi_max(m);
            win = I(i_win);

            % Agents incur cost equal to their urgency, except passing agent
            c_bid_all_u(t,I) = u;
            c_bid_all_u(t,win) = 0;

            % Get delayed agents. They will be getting karma
            lose = func.get_lose(I, i_win);

            % Update karma
            if t < param.tot_num_inter
                [k_win, k_lose] = func.get_karma_payments(m_win, lose, k_bid_all_u(t,:), param);
                k_bid_all_u(t+1,:) = k_bid_all_u(t,:);
                k_bid_all_u(t+1,win) = k_bid_all_u(t+1,win) - k_win;
                k_bid_all_u(t+1,lose) = k_bid_all_u(t+1,lose) + k_lose;
            end
        end
        
        %% Nash equilibrium karma policies
        if control.karma_ne_policies
            for i_alpha = 1 : param.num_alpha
                % Get agents' bids from their policies
                k = k_ne{i_alpha}(t,I);
                m = zeros(1, param.I_size);
                for i_agent = 1 : param.I_size
                    i_u = find(param.U == u(i_agent));
                    i_k = find(param.K == k(i_agent));
                    m(i_agent) = datasample(param.M, 1, 'Weights', squeeze(pi_ne{i_alpha}(i_u,i_k,:)));
                end
                
                % Agent bidding max karma passes and pays karma bidded
                [m_win, i_win] = func.multi_max(m);
                win = I(i_win);

                % Agents incur cost equal to their urgency, except passing agent
                c_ne{i_alpha}(t,I) = u;
                c_ne{i_alpha}(t,win) = 0;

                % Get delayed agents. They will be getting karma
                lose = func.get_lose(I, i_win);

                % Update karma
                if t < param.tot_num_inter
                    [k_win, k_lose] = func.get_karma_payments(m_win, lose, k_ne{i_alpha}(t,:), param);
                    k_ne{i_alpha}(t+1,:) = k_ne{i_alpha}(t,:);
                    k_ne{i_alpha}(t+1,win) = k_ne{i_alpha}(t+1,win) - k_win;
                    k_ne{i_alpha}(t+1,lose) = k_ne{i_alpha}(t+1,lose) + k_lose;
                end
            end
        end
        %% Social welfare karma policy
        if control.karma_sw_policy
            % Get agents' bids from their policies
            k = k_sw(t,I);
            m = zeros(1, param.I_size);
            for i_agent = 1 : param.I_size
                i_u = find(param.U == u(i_agent));
                i_k = find(param.K == k(i_agent));
                m(i_agent) = datasample(param.M, 1, 'Weights', squeeze(pi_sw(i_u,i_k,:)));
            end

            % Agent bidding max karma wins and pays karma bidded
            [m_win, i_win] = func.multi_max(m);
            win = I(i_win);

            % Agents incur cost equal to their urgency, except winning agent
            c_sw(t,I) = u;
            c_sw(t,win) = 0;

            % Get losing agents. They will be getting karma
            lose = func.get_lose(I, i_win);

            % Update karma
            if t < param.tot_num_inter
                [k_win, k_lose] = func.get_karma_payments(m_win, lose, k_sw(t,:), param);
                k_sw(t+1,:) = k_sw(t,:);
                k_sw(t+1,win) = k_sw(t+1,win) - k_win;
                k_sw(t+1,lose) = k_sw(t+1,lose) + k_lose;
            end
        end
    end
end

%% Perfromance measures
% Accumulated costs per agent at each time step
a_rand = func.get_accumulated_cost(c_rand, param);
a_1 = func.get_accumulated_cost(c_1, param);
a_2 = func.get_accumulated_cost(c_2, param);
a_1_2 = func.get_accumulated_cost(c_1_2, param);
if control.lim_mem_policies
    a_lim_mem = cell(param.num_lim_mem_steps, 1);
    a_lim_mem_u = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        a_lim_mem{i_lim_mem} = func.get_accumulated_cost(c_lim_mem{i_lim_mem}, param);
        a_lim_mem_u{i_lim_mem} = func.get_accumulated_cost(c_lim_mem_u{i_lim_mem}, param);
    end
end
if control.karma_heuristic_policies
    a_bid_1 = func.get_accumulated_cost(c_bid_1, param);
    a_bid_1_u = func.get_accumulated_cost(c_bid_1_u, param);
    a_bid_all = func.get_accumulated_cost(c_bid_all, param);
    a_bid_all_u = func.get_accumulated_cost(c_bid_all_u, param);
end
if control.karma_ne_policies
    a_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        a_ne{i_alpha} = func.get_accumulated_cost(c_ne{i_alpha}, param);
    end
end
if control.karma_sw_policy
    a_sw = func.get_accumulated_cost(c_sw, param);
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
        for i_lim_mem = 1 : param.num_lim_mem_steps
            a_lim_mem{i_lim_mem} = a_lim_mem{i_lim_mem}(actual_t,:);
            a_lim_mem_u{i_lim_mem} = a_lim_mem_u{i_lim_mem}(actual_t,:);
        end
    end
    if control.karma_heuristic_policies
        a_bid_1 = a_bid_1(actual_t,:);
        a_bid_1_u = a_bid_1_u(actual_t,:);
        a_bid_all = a_bid_all(actual_t,:);
        a_bid_all_u = a_bid_all_u(actual_t,:);
    end
    if control.karma_ne_policies
        for i_alpha = 1 : param.num_alpha
            a_ne{i_alpha} = a_ne{i_alpha}(actual_t,:);
        end
    end
    if control.sw_policy
        a_sw = a_sw(actual_t,:);
    end
end

% Accumulated costs per agent at each time step, normalized by their
% respective number of interactions. Only compute if normalization flag is
% on
if param.normalize_cost
    % Zeros in number of interactions are replaces by 1 to avoid division by 0
    num_inter_div = num_inter;
    num_inter_div(num_inter_div == 0) = 1;
    a_rand = a_rand ./ num_inter_div;
    a_1 = a_1 ./ num_inter_div;
    a_2 = a_2 ./ num_inter_div;
    a_1_2 = a_1_2 ./ num_inter_div;
    if control.lim_mem_policies
        for i_lim_mem = 1 : param.num_lim_mem_steps
            a_lim_mem{i_lim_mem} = a_lim_mem{i_lim_mem} ./ num_inter_div;
            a_lim_mem_u{i_lim_mem} = a_lim_mem_u{i_lim_mem} ./ num_inter_div;
        end
    end
    if control.karma_heuristic_policies
        a_bid_1 = a_bid_1 ./ num_inter_div;
        a_bid_1_u = a_bid_1_u ./ num_inter_div;
        a_bid_all = a_bid_all ./ num_inter_div;
        a_bid_all_u = a_bid_all_u ./ num_inter_div;
    end
    if control.karma_ne_policies
        for i_alpha = 1 : param.num_alpha
            a_ne{i_alpha} = a_ne{i_alpha} ./ num_inter_div;
        end
    end
    if control.karma_sw_policy
        a_sw = a_sw ./ num_inter_div;
    end
end

% Inefficiency vs. time
W1_rand = mean(a_rand, 2);
W1_1 = mean(a_1, 2);
W1_2 = mean(a_2, 2);
W1_1_2 = mean(a_1_2, 2);
if control.lim_mem_policies
    W1_lim_mem = cell(param.num_lim_mem_steps, 1);
    W1_lim_mem_u = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        W1_lim_mem{i_lim_mem} = mean(a_lim_mem{i_lim_mem}, 2);
        W1_lim_mem_u{i_lim_mem} = mean(a_lim_mem_u{i_lim_mem}, 2);
    end
end
if control.karma_heuristic_policies
    W1_bid_1 = mean(a_bid_1, 2);
    W1_bid_1_u = mean(a_bid_1_u, 2);
    W1_bid_all = mean(a_bid_all, 2);
    W1_bid_all_u = mean(a_bid_all_u, 2);
end
if control.karma_ne_policies
    W1_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        W1_ne{i_alpha} = mean(a_ne{i_alpha}, 2);
    end
end
if control.karma_sw_policy
    W1_sw = mean(a_sw, 2);
end

% Unfairness vs. time
W2_rand = var(a_rand, [], 2);
W2_1 = var(a_1, [], 2);
W2_2 = var(a_2, [], 2);
W2_1_2 = var(a_1_2, [], 2);
if control.lim_mem_policies
    W2_lim_mem = cell(param.num_lim_mem_steps, 1);
    W2_lim_mem_u = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        W2_lim_mem{i_lim_mem} = var(a_lim_mem{i_lim_mem}, [], 2);
        W2_lim_mem_u{i_lim_mem} = var(a_lim_mem_u{i_lim_mem}, [], 2);
    end
end
if control.karma_heuristic_policies
    W2_bid_1 = var(a_bid_1, [], 2);
    W2_bid_1_u = var(a_bid_1_u, [], 2);
    W2_bid_all = var(a_bid_all, [], 2);
    W2_bid_all_u = var(a_bid_all_u, [], 2);
end
if control.karma_ne_policies
    W2_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        W2_ne{i_alpha} = var(a_ne{i_alpha}, [], 2);
    end
end
if control.karma_sw_policy
    W2_sw = var(a_sw, [], 2);
end

% Standardized accumulated costs. Standardization method is a parameter.
% Required for autocorrelation and allows to investigate 'mixing' ability
% of policies
switch param.standardization_method
    % 0-mean 1-variance standardization
    case 0
        a_rand_std = func.standardize_mean_var(a_rand, W1_rand, W2_rand);
        a_1_std = func.standardize_mean_var(a_1, W1_1, W2_1);
        a_2_std = func.standardize_mean_var(a_2, W1_2, W2_2);
        a_1_2_std = func.standardize_mean_var(a_1_2, W1_1_2, W2_1_2);
        if control.lim_mem_policies
            a_lim_mem_std = cell(param.num_lim_mem_steps, 1);
            a_lim_mem_u_std = cell(param.num_lim_mem_steps, 1);
            for i_lim_mem = 1 : param.num_lim_mem_steps
                a_lim_mem_std{i_lim_mem} = func.standardize_mean_var(a_lim_mem{i_lim_mem}, W1_lim_mem{i_lim_mem}, W2_lim_mem{i_lim_mem});
                a_lim_mem_u_std{i_lim_mem} = func.standardize_mean_var(a_lim_mem_u{i_lim_mem}, W1_lim_mem_u{i_lim_mem}, W2_lim_mem_u{i_lim_mem});
            end
        end
        if control.karma_heuristic_policies
            a_bid_1_std = func.standardize_mean_var(a_bid_1, W1_bid_1, W2_bid_1);
            a_bid_1_u_std = func.standardize_mean_var(a_bid_1_u, W1_bid_1_u, W2_bid_1_u);
            a_bid_all_std = func.standardize_mean_var(a_bid_all, W1_bid_all, W2_bid_all);
            a_bid_all_u_std = func.standardize_mean_var(a_bid_all, W1_bid_all_u, W2_bid_all_u);
        end
        if control.karma_ne_policies
            a_ne_std = cell(param.num_alpha, 1);
            for i_alpha = 1 : param.num_alpha
                a_ne_std{i_alpha} = func.standardize_mean_var(a_ne{i_alpha}, W1_ne{i_alpha}, W2_ne{i_alpha});
            end
        end
        if control.karma_sw_policy
            a_sw_std = func.standardize_mean_var(a_sw, W1_sw, W2_sw);
        end
    % Order ranking standardization
    case 1
        a_rand_std = func.order_rank(a_rand);
        a_1_std = func.order_rank(a_1);
        a_2_std = func.order_rank(a_2);
        a_1_2_std = func.order_rank(a_1_2);
        if control.lim_mem_policies
            a_lim_mem_std = cell(param.num_lim_mem_steps, 1);
            a_lim_mem_u_std = cell(param.num_lim_mem_steps, 1);
            for i_lim_mem = 1 : param.num_lim_mem_steps
                a_lim_mem_std{i_lim_mem} = func.order_rank(a_lim_mem{i_lim_mem});
                a_lim_mem_u_std{i_lim_mem} = func.order_rank(a_lim_mem_u{i_lim_mem});
            end
        end
        if control.karma_heuristic_policies
            a_bid_1_std = func.order_rank(a_bid_1);
            a_bid_1_u_std = func.order_rank(a_bid_1_u);
            a_bid_all_std = func.order_rank(a_bid_all);
            a_bid_all_u_std = func.order_rank(a_bid_all);
        end
        if control.karma_ne_policies
            a_ne_std = cell(param.num_alpha, 1);
            for i_alpha = 1 : param.num_alpha
                a_ne_std{i_alpha} = func.order_rank(a_ne{i_alpha});
            end
        end
        if control.karma_sw_policy
            a_sw_std = func.order_rank(a_sw);
        end
    % normalized order ranking standardization, i.e. order ranking scaled
    % between 0-1
    case 2
        a_rand_std = func.order_rank_norm(a_rand);
        a_1_std = func.order_rank_norm(a_1);
        a_2_std = func.order_rank_norm(a_2);
        a_1_2_std = func.order_rank_norm(a_1_2);
        if control.lim_mem_policies
            a_lim_mem_std = cell(param.num_lim_mem_steps, 1);
            a_lim_mem_u_std = cell(param.num_lim_mem_steps, 1);
            for i_lim_mem = 1 : param.num_lim_mem_steps
                a_lim_mem_std{i_lim_mem} = func.order_rank_norm(a_lim_mem{i_lim_mem});
                a_lim_mem_u_std{i_lim_mem} = func.order_rank_norm(a_lim_mem_u{i_lim_mem});
            end
        end
        if control.karma_heuristic_policies
            a_bid_1_std = func.order_rank_norm(a_bid_1);
            a_bid_1_u_std = func.order_rank_norm(a_bid_1_u);
            a_bid_all_std = func.order_rank_norm(a_bid_all);
            a_bid_all_u_std = func.order_rank_norm(a_bid_all);
        end
        if control.karma_ne_policies
            a_ne_std = cell(param.num_alpha, 1);
            for i_alpha = 1 : param.num_alpha
                a_ne_std{i_alpha} = func.order_rank_norm(a_ne{i_alpha});
            end
        end
        if control.karma_sw_policy
            a_sw_std = func.order_rank_norm(a_sw);
        end
end

%% Autocorrelation of accumulated cost
% Provides indication on how well population cost 'mixes' with time
if control.compute_autocorrelation
    fprintf('Computing autocorrelation for baseline-random\n');
    [a_rand_acorr, acorr_tau] = func.autocorrelation(a_rand_std);
    fprintf('Computing autocorrelation for centralized-urgency\n');
    a_1_acorr = func.autocorrelation(a_1_std);
    fprintf('Computing autocorrelation for centralized-cost\n');
    a_2_acorr = func.autocorrelation(a_2_std);
    fprintf('Computing autocorrelation for centralized-urgency-then-cost\n');
    a_1_2_acorr = func.autocorrelation(a_1_2_std);
    if control.lim_mem_policies
        a_lim_mem_acorr = cell(param.num_lim_mem_steps, 1);
        a_lim_mem_u_acorr = cell(param.num_lim_mem_steps, 1);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            fprintf('Computing autocorrelation for centralized-cost-mem-%d\n', param.lim_mem_steps(i_lim_mem));
            a_lim_mem_acorr{i_lim_mem} = func.autocorrelation(a_lim_mem_std{i_lim_mem});
            fprintf('Computing autocorrelation for centralized-urgency-then-cost-mem-%d\n', param.lim_mem_steps(i_lim_mem));
            a_lim_mem_u_acorr{i_lim_mem} = func.autocorrelation(a_lim_mem_u_std{i_lim_mem});
        end
    end
    if control.karma_heuristic_policies
        fprintf('Computing autocorrelation for bid-1-always\n');
        a_bid_1_acorr = func.autocorrelation(a_bid_1_std);
        fprintf('Computing autocorrelation for bid-1-if-urgent\n');
        a_bid_1_u_acorr = func.autocorrelation(a_bid_1_u_std);
        fprintf('Computing autocorrelation for bid-all-always\n');
        a_bid_all_acorr = func.autocorrelation(a_bid_all_std);
        fprintf('Computing autocorrelation for bid-all-if-urgent\n');
        a_bid_all_u_acorr = func.autocorrelation(a_bid_all_u_std);
    end
    if control.karma_ne_policies
        a_ne_acorr = cell(param.num_alpha, 1);
        for i_alpha = 1 : param.num_alpha
            fprintf('Computing autocorrelation for alpha-%f\n', param.alpha(i_alpha));
            a_ne_acorr{i_alpha} = func.autocorrelation(a_ne_std{i_alpha});
        end
    end
    if control.karma_sw_policy
        fprintf('Computing autocorrelation for social-welfare\n');
        a_sw_acorr = func.autocorrelation(a_sw_std);
    end
end

%% Plots
fprintf('Plotting\n');

%% Scatter plot - Inefficiency vs unfairness
figure(fg);
fg = fg + 1;
fig = gcf;
fig.Position = [0, 0, screenwidth, screenheight];
pl = plot(W1_rand(end), W2_rand(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = "baseline-random";
hold on;
pl = plot(W1_1(end), W2_1(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = [lgd_text, "centralized-urgency"];
pl = plot(W1_2(end), W2_2(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = [lgd_text, "centralized-cost"];
pl = plot(W1_1_2(end), W2_1_2(end),...
    'LineStyle', 'none',...
    'Marker', 'p',...
    'MarkerSize', 10);
pl.MarkerFaceColor = pl.Color;
lgd_text = [lgd_text, "centralized-urgency-then-cost"];
if control.lim_mem_policies
    for i_lim_mem = 1 : param.num_lim_mem_steps
        pl = plot(W1_lim_mem{i_lim_mem}(end), W2_lim_mem{i_lim_mem}(end),...
            'LineStyle', 'none',...
            'Marker', '*',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        lgd_text = [lgd_text, strcat("centralized-cost-mem-", int2str(param.lim_mem_steps(i_lim_mem)))];
    end
    for i_lim_mem = 1 : param.num_lim_mem_steps
        pl = plot(W1_lim_mem_u{i_lim_mem}(end), W2_lim_mem{i_lim_mem}(end),...
            'LineStyle', 'none',...
            'Marker', '*',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        lgd_text = [lgd_text, strcat("centralized-urgency-then-cost-mem-", int2str(param.lim_mem_steps(i_lim_mem)))];
    end
end
if control.karma_heuristic_policies
    pl = plot(W1_bid_1(end), W2_bid_1(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "bid-1-always"];
    pl = plot(W1_bid_1_u(end), W2_bid_1_u(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "bid-1-if-urgent"];
    pl = plot(W1_bid_all(end), W2_bid_all(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "bid-all-always"];
    pl = plot(W1_bid_all_u(end), W2_bid_all_u(end),...
        'LineStyle', 'none',...
        'Marker', 'd',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "bid-all-if-urgent"];
end
if control.karma_ne_policies
    for i_alpha = 1 : param.num_alpha
        pl = plot(W1_ne{i_alpha}(end), W2_ne{i_alpha}(end),...
            'LineStyle', 'none',...
            'Marker', 'o',...
            'MarkerSize', 10);
        pl.MarkerFaceColor = pl.Color;
        lgd_text = [lgd_text, strcat("$\alpha$ = ", num2str(param.alpha(i_alpha), '%.2f'))];
    end
end
if control.karma_sw_policy
    pl = plot(W1_sw(end), W2_sw(end),...
        'LineStyle', 'none',...
        'Marker', 's',...
        'MarkerSize', 10);
    pl.MarkerFaceColor = pl.Color;
    lgd_text = [lgd_text, "social-welfare"];
end
axes = gca;
func.axis_semi_tight(axes, 1.2);
axes.Title.Interpreter = 'latex';
axes.Title.String = 'Performance Comparison';
axes.Title.FontSize = 18;
axes.XAxis.TickLabelInterpreter = 'latex';
axes.XAxis.FontSize = 10;
axes.YAxis.TickLabelInterpreter = 'latex';
axes.YAxis.FontSize = 10;
axes.XLabel.Interpreter = 'latex';
axes.XLabel.String = 'Efficiency (mean of cost)';
axes.XLabel.FontSize = 14;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = 'Fairness (variance of cost)';
axes.YLabel.FontSize = 14;
lgd = legend(lgd_text);
lgd.Interpreter = 'latex';
lgd.FontSize = 12;
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
    num_cols = round(sqrt(screenwidth / screenheight * param.num_lim_mem_steps));
    num_rows = ceil(param.num_lim_mem_steps / num_cols);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        subplot(num_rows,num_cols,i_lim_mem);
        plot(a_lim_mem{i_lim_mem});
        hold on;
        plot(W1_lim_mem{i_lim_mem}, 'Linewidth', 3);
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = ['centralized-cost-mem-', int2str(param.lim_mem_steps(i_lim_mem))];
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
    for i_lim_mem = 1 : param.num_lim_mem_steps
        plot(W2_lim_mem{i_lim_mem}, '--');
    end
    for i_lim_mem = 1 : param.num_lim_mem_steps
        plot(W2_lim_mem_u{i_lim_mem}, '--');
    end
end
if control.karma_heuristic_policies
    plot(W2_bid_1, '-.');
    plot(W2_bid_1_u, '-.');
    plot(W2_bid_all, '-.');
    plot(W2_bid_all_u, '-.');
end
if control.karma_ne_policies
    for i_alpha = 1 : param.num_alpha
        plot(W2_ne{i_alpha}, ':');
    end
end
if control.karma_sw_policy
    plot(W2_sw, ':', 'LineWidth', 2);
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
    for i_lim_mem = 1 : param.num_lim_mem_steps
        subplot(num_rows,num_cols,i_lim_mem);
        plot(a_lim_mem_std{i_lim_mem});
        axes = gca;
        axis tight;
        axes.Title.Interpreter = 'latex';
        axes.Title.String = ['centralized-cost-mem-', int2str(param.lim_mem_steps(i_lim_mem))];
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
    plot(acorr_tau, a_rand_acorr, 'LineWidth', 2);
    hold on;
    plot(acorr_tau, a_1_acorr, 'LineWidth', 2);
    plot(acorr_tau, a_2_acorr, 'LineWidth', 2);
    plot(acorr_tau, a_1_2_acorr, 'LineWidth', 2);
    if control.lim_mem_policies
        for i_lim_mem = 1 : param.num_lim_mem_steps
            plot(acorr_tau, a_lim_mem_acorr{i_lim_mem}, '--');
        end
        for i_lim_mem = 1 : param.num_lim_mem_steps
            plot(acorr_tau, a_lim_mem_u_acorr{i_lim_mem}, '--');
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
            plot(acorr_tau, a_ne_acorr{i_alpha}, ':');
        end
    end
    if control.karma_sw_policy
        plot(acorr_tau, a_sw_acorr, ':', 'LineWidth', 2);
    end
    axis tight;
    axes = gca;
    yl = ylim(axes);
    stem(0, a_rand_acorr(acorr_tau == 0));
    ylim(axes, yl);
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

%% Inform user when done
fprintf('DONE\n\n');