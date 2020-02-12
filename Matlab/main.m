clear;
close all;
clc;

%% Control randomization
rng(0);
[u_stream, agent_stream, karma_stream] = RandStream.create('mrg32k3a', 'NumStreams', 3);

%% Code control bits
% Autocorrelation takes long time to compute
control.compute_a_acorr = false;

% Flag to simulate centralized limited memory policies
control.lim_mem_policies = false;

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
    c_lim_mem_u = cell(param.num_lim_mem_steps, 1);
    c_in_mem_u  = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
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
    % Cost for bid random always policy
    c_bid_rand = zeros(param.tot_num_inter, param.N);
    % Cost for bid random if urgent policy
    c_bid_rand_u = zeros(param.tot_num_inter, param.N);
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
% Initial karma, for cases when it is common amongst multiple policies
if control.karma_heuristic_policies || (control.karma_ne_policies || control.karma_sw_policy) && param.karma_initialization ~= 2
    if param.karma_initialization == 0
        init_k = param.k_ave * ones(1, param.N);
    else
        s_up_k_uniform = func.get_s_up_k_uniform(param);
        init_k = func.get_init_k(s_up_k_uniform, param.K, param);
    end
end
    
% Heuristic karma policies
if control.karma_heuristic_policies
    % Karma for bid 1 always policy
    k_bid_1 = zeros(param.tot_num_inter, param.N);
    k_bid_1(1,:) = init_k;
    % Karma for bid 1 if urgent policy
    k_bid_1_u = zeros(param.tot_num_inter, param.N);
    k_bid_1_u(1,:) = init_k;
    % Karma for bid random always policy
    k_bid_rand = zeros(param.tot_num_inter, param.N);
    k_bid_rand(1,:) = init_k;
    % Karma for bid random if urgent policy
    k_bid_rand_u = zeros(param.tot_num_inter, param.N);
    k_bid_rand_u(1,:) = init_k;
end

% Nash equilibrium karma policies
if control.karma_ne_policies
    k_ne = cell(param.num_alpha, 1);
    if param.karma_initialization == 2
        file_str = ['karma_nash_equilibrium/results/k_max_', num2str(param.k_max, '%02d'), '_k_ave_', num2str(param.k_ave, '%02d'), '/alpha_'];
    end
    for i_alpha = 1 : param.num_alpha
        k_ne{i_alpha} = zeros(param.tot_num_inter, param.N);
        
        if param.karma_initialization == 2
            % Initialize karma as per stationary distribution predicted by NE
            % algorithm
            load([file_str, num2str(param.alpha(i_alpha), '%.2f'), '.mat'], 'ne_d_up_u_k');
            k_ne{i_alpha}(1,:) = func.get_init_k(sum(ne_d_up_u_k), param.K, param);
%             load([file_str, num2str(param.alpha(i_alpha), '%.2f'), '.mat'], 'D_up_u_k');
%             k_ne{i_alpha}(1,:) = func.get_init_k(sum(D_up_u_k), param.K, param);
        else
            k_ne{i_alpha}(1,:) = init_k;
        end
    end
end

% Social welfare karma policy
if control.karma_sw_policy
    k_sw = zeros(param.tot_num_inter, param.N);
    
    if param.karma_initialization == 2
        % Initialize karma as per stationary distribution predicted by SW
        % algorithm
        load(['karma_nash_equilibrium/results/sw_k_max_', num2str(param.k_max, '%02d'), '/k_ave_', num2str(param.k_ave, '%02d'), '.mat'], 'sw_d_up_u_k');
        k_sw(1,:) = func.get_init_k(sum(sw_d_up_u_k), param.K, param);
    else
        k_sw(1,:) = init_k;
    end
end

%% Policy matrices for different karma policies
% Nash equilibrium karma policies
if control.karma_ne_policies
    pi_ne = cell(param.num_alpha, 1);
    file_str = ['karma_nash_equilibrium/results/k_max_', num2str(param.k_max, '%02d'), '_k_ave_', num2str(param.k_ave, '%02d'), '/alpha_'];
    for i_alpha = 1 : param.num_alpha
        load([file_str, num2str(param.alpha(i_alpha), '%.2f'), '.mat'], 'ne_pi_down_u_k_up_m');
        pi_ne{i_alpha} = ne_pi_down_u_k_up_m;
%         load([file_str, num2str(param.alpha(i_alpha), '%.2f'), '.mat'], 'pi_down_u_k_up_m');
%         pi_ne{i_alpha} = pi_down_u_k_up_m;
        % Eliminate the possibility of very unlikely messages, which have
        % non-zero probability due to algorithm numerics
        pi_ne{i_alpha}(pi_ne{i_alpha} < 1e-3) = 0;
    end
end

% Social welfare karma policy
if control.karma_sw_policy
    load(['karma_nash_equilibrium/results/sw_k_max_', num2str(param.k_max, '%02d'), '/k_ave_', num2str(param.k_ave, '%02d'), '.mat'], 'sw_pi_down_u_k_up_m');
    pi_sw = sw_pi_down_u_k_up_m;
    % Eliminate the possibility of very unlikely messages, which have
    % non-zero probability due to algorithm numerics
    pi_sw(pi_sw < 1e-3) = 0;
end

%% Number of times each agent was in an intersection, as an accumulated sum
num_inter = zeros(param.tot_num_inter, param.N);

%% Simulation run
% Convention:   win := agents that win
%               lose := agent(s) that lose
for day = 1 : param.num_days
    % Pick urgency in {0,U} uniformly at random for all agents. Urgency
    % stays constant for agents per day
    if param.num_inter_per_day > 1
        u_today = datasample(u_stream, param.U, param.N).';
    end
    
    for inter = 1 : param.num_inter_per_day
        t = (day - 1) * param.num_inter_per_day + inter;
        % Tell user where we are
        fprintf('Day: %d Interaction: %d Timestep: %d\n', day, inter, t);

        if ~param.same_num_inter
            % Sample agents i & j uniformly at random
            I = datasample(agent_stream, population, param.I_size, 'Replace', false);
        else
            % If all population has been sampled, re-fill population
            if isempty(population)
                population = 1 : param.N;
            end
            % Sample agents i & j uniformly at random and remove them from
            % population
            I = datasample(agent_stream, population, param.I_size, 'Replace', false);
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
        if param.num_inter_per_day > 1
            u = u_today(I);
        else
            u = datasample(u_stream, param.U, param.I_size).';
        end

        %% Random policy
        % Simply choose 'first' agent, since there is already randomization
        % in the agent order
        win = I(1);

        % Agents incur cost equal to their urgency, except passing agent
        c_rand(t,I) = u;
        c_rand(t,win) = 0;

        %% CENTRALIZED POLICIES %%
        %% Centralized policy 1 - minimize W1, coin-flip if tie
        % Find agent(s) with max urgency, which are candidates for passing
        [~, i_win] = func.multi_maxes(u);
        win_max_u = I(i_win);
        num_max_u = length(win_max_u);
        
        % Choose 'first' urgent agent (there is already randomization in
        % the agent order)
        win = win_max_u(1);

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
        [~, i_win] = max(a_u);
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
            [~, i_win] = max(a_u);
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
                    [~, i_win] = max(a_u);
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
        
        %% Prepare random stream for karma policies
        karma_stream_state = karma_stream.State;
        
        %% HEURISTIC KARMA POLICIES
        if control.karma_heuristic_policies
            %% Bid 1 always policy
            % Agents simply bid 1, if they have it
            m = min([ones(1, param.I_size); k_bid_1(t,I)]);

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = max(m);
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
            m = min([ones(1, param.I_size); k_bid_1_u(t,I)]);
            m(u == 0) = 0;

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = max(m);
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

            %% Bid random always policy
            % Agents simply bid random
            k = k_bid_rand(t,I);
            m = zeros(1, param.I_size);
            for i_agent = 1 : param.I_size
                m(i_agent) = datasample(karma_stream, 0 : k(i_agent), 1);
            end

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = max(m);
            win = I(i_win);

            % Agents incur cost equal to their urgency, except passing agent
            c_bid_rand(t,I) = u;
            c_bid_rand(t,win) = 0;

            % Get delayed agents. They will be getting karma
            lose = func.get_lose(I, i_win);

            % Update karma
            if t < param.tot_num_inter
                [k_win, k_lose] = func.get_karma_payments(m_win, lose, k_bid_rand(t,:), param);
                k_bid_rand(t+1,:) = k_bid_rand(t,:);
                k_bid_rand(t+1,win) = k_bid_rand(t+1,win) - k_win;
                k_bid_rand(t+1,lose) = k_bid_rand(t+1,lose) + k_lose;
            end

            %% Bid random if urgent policy
            % Agents bid random if urgent, greater than zero if they can
            k = k_bid_rand(t,I);
            m = zeros(1, param.I_size);
            karma_stream.State = karma_stream_state;
            for i_agent = 1 : param.I_size
                if u(i_agent) ~= 0 && k(i_agent) > 0
                    m(i_agent) = datasample(karma_stream, 1 : k(i_agent), 1);
                end
            end

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = max(m);
            win = I(i_win);

            % Agents incur cost equal to their urgency, except passing agent
            c_bid_rand_u(t,I) = u;
            c_bid_rand_u(t,win) = 0;

            % Get delayed agents. They will be getting karma
            lose = func.get_lose(I, i_win);

            % Update karma
            if t < param.tot_num_inter
                [k_win, k_lose] = func.get_karma_payments(m_win, lose, k_bid_rand_u(t,:), param);
                k_bid_rand_u(t+1,:) = k_bid_rand_u(t,:);
                k_bid_rand_u(t+1,win) = k_bid_rand_u(t+1,win) - k_win;
                k_bid_rand_u(t+1,lose) = k_bid_rand_u(t+1,lose) + k_lose;
            end
        end
        
        %% Nash equilibrium karma policies
        if control.karma_ne_policies
            for i_alpha = 1 : param.num_alpha
                % Get agents' bids from their policies
                k = k_ne{i_alpha}(t,I);
                m = zeros(1, param.I_size);
                karma_stream.State = karma_stream_state;
                for i_agent = 1 : param.I_size
                    i_u = find(param.U == u(i_agent));
                    i_k = find(param.K == k(i_agent));
                    m(i_agent) = datasample(karma_stream, param.M, 1, 'Weights', squeeze(pi_ne{i_alpha}(i_u,i_k,:)));
%                     % TEST CODE %
%                     if param.alpha(i_alpha) == 0 && u(i_agent) ~= 0 && m(i_agent) ~= k(i_agent)
%                         fprintf('DEBUG u = %d k = %d m = %d\n', u(i_agent), k(i_agent), m(i_agent));
%                         pause;
%                     end
                end
                
                % Agent bidding max karma passes and pays karma bidded
                [m_win, i_win] = max(m);
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
            karma_stream.State = karma_stream_state;
            for i_agent = 1 : param.I_size
                i_u = find(param.U == u(i_agent));
                i_k = find(param.K == k(i_agent));
                m(i_agent) = datasample(karma_stream, param.M, 1, 'Weights', squeeze(pi_sw(i_u,i_k,:)));
            end

            % Agent bidding max karma wins and pays karma bidded
            [m_win, i_win] = max(m);
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
fprintf('Computing performance measures\n');

% Accumulated costs per agent at each time step
fprintf('Computing accumulated costs\n');
a_rand = func.get_accumulated_cost(c_rand, param);
clear c_rand;
a_1 = func.get_accumulated_cost(c_1, param);
clear c_1;
a_2 = func.get_accumulated_cost(c_2, param);
clear c_2;
a_1_2 = func.get_accumulated_cost(c_1_2, param);
clear c_1_2;
if control.lim_mem_policies
    a_lim_mem_u = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        a_lim_mem_u{i_lim_mem} = func.get_accumulated_cost(c_lim_mem_u{i_lim_mem}, param);
        c_lim_mem_u{i_lim_mem} = [];
    end
    clear c_lim_mem_u;
end
if control.karma_heuristic_policies
    a_bid_1 = func.get_accumulated_cost(c_bid_1, param);
    clear c_bid_1;
    a_bid_1_u = func.get_accumulated_cost(c_bid_1_u, param);
    clear c_bid_1_u;
    a_bid_rand = func.get_accumulated_cost(c_bid_rand, param);
    clear c_bid_rand;
    a_bid_rand_u = func.get_accumulated_cost(c_bid_rand_u, param);
    clear c_bid_rand_u;
end
if control.karma_ne_policies
    a_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        a_ne{i_alpha} = func.get_accumulated_cost(c_ne{i_alpha}, param);
        c_ne{i_alpha} = [];
    end
    clear c_ne;
end
if control.karma_sw_policy
    a_sw = func.get_accumulated_cost(c_sw, param);
    clear c_sw;
end

% If number of interactions per agent is fixed, true time is interprated as
% the time after which all agents have participated in an interaction
if param.same_num_inter
    fprintf('Collapsing time to instances where all agents have participated in an interaction\n');
    actual_t = param.num_inter_in_N : param.num_inter_in_N : param.tot_num_inter;
    num_inter = num_inter(actual_t,:);
    a_rand = a_rand(actual_t,:);
    a_1 = a_1(actual_t,:);
    a_2 = a_2(actual_t,:);
    a_1_2 = a_1_2(actual_t,:);
    if control.lim_mem_policies
        for i_lim_mem = 1 : param.num_lim_mem_steps
            a_lim_mem_u{i_lim_mem} = a_lim_mem_u{i_lim_mem}(actual_t,:);
        end
    end
    if control.karma_heuristic_policies
        a_bid_1 = a_bid_1(actual_t,:);
        a_bid_1_u = a_bid_1_u(actual_t,:);
        a_bid_rand = a_bid_rand(actual_t,:);
        a_bid_rand_u = a_bid_rand_u(actual_t,:);
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
    fprintf('Normalizing accumulated costs\n');
    % Zeros in number of interactions are replaces by 1 to avoid division by 0
    num_inter_div = num_inter;
    num_inter_div(num_inter_div == 0) = 1;
    a_rand = a_rand ./ num_inter_div;
    a_1 = a_1 ./ num_inter_div;
    a_2 = a_2 ./ num_inter_div;
    a_1_2 = a_1_2 ./ num_inter_div;
    if control.lim_mem_policies
        for i_lim_mem = 1 : param.num_lim_mem_steps
            a_lim_mem_u{i_lim_mem} = a_lim_mem_u{i_lim_mem} ./ num_inter_div;
        end
    end
    if control.karma_heuristic_policies
        a_bid_1 = a_bid_1 ./ num_inter_div;
        a_bid_1_u = a_bid_1_u ./ num_inter_div;
        a_bid_rand = a_bid_rand ./ num_inter_div;
        a_bid_rand_u = a_bid_rand_u ./ num_inter_div;
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
fprintf('Computing efficiencies\n');
W1_rand = mean(a_rand, 2);
W1_1 = mean(a_1, 2);
W1_2 = mean(a_2, 2);
W1_1_2 = mean(a_1_2, 2);
if control.lim_mem_policies
    W1_lim_mem_u = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        W1_lim_mem_u{i_lim_mem} = mean(a_lim_mem_u{i_lim_mem}, 2);
    end
end
if control.karma_heuristic_policies
    W1_bid_1 = mean(a_bid_1, 2);
    W1_bid_1_u = mean(a_bid_1_u, 2);
    W1_bid_rand = mean(a_bid_rand, 2);
    W1_bid_rand_u = mean(a_bid_rand_u, 2);
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
fprintf('Computing fairness\n');
W2_rand = var(a_rand, [], 2);
W2_1 = var(a_1, [], 2);
W2_2 = var(a_2, [], 2);
W2_1_2 = var(a_1_2, [], 2);
if control.lim_mem_policies
    W2_lim_mem_u = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        W2_lim_mem_u{i_lim_mem} = var(a_lim_mem_u{i_lim_mem}, [], 2);
    end
end
if control.karma_heuristic_policies
    W2_bid_1 = var(a_bid_1, [], 2);
    W2_bid_1_u = var(a_bid_1_u, [], 2);
    W2_bid_rand = var(a_bid_rand, [], 2);
    W2_bid_rand_u = var(a_bid_rand_u, [], 2);
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
% of policies. Only compute if autocorrelation flag is on
if control.compute_a_acorr
    fprintf('Standardizing accumulated costs\n');
    switch param.standardization_method
        % 0-mean 1-variance standardization
        case 0
            a_rand_std = func.standardize_mean_var(a_rand, W1_rand, W2_rand);
            a_1_std = func.standardize_mean_var(a_1, W1_1, W2_1);
            a_2_std = func.standardize_mean_var(a_2, W1_2, W2_2);
            a_1_2_std = func.standardize_mean_var(a_1_2, W1_1_2, W2_1_2);
            if control.lim_mem_policies
                a_lim_mem_u_std = cell(param.num_lim_mem_steps, 1);
                for i_lim_mem = 1 : param.num_lim_mem_steps
                    a_lim_mem_u_std{i_lim_mem} = func.standardize_mean_var(a_lim_mem_u{i_lim_mem}, W1_lim_mem_u{i_lim_mem}, W2_lim_mem_u{i_lim_mem});
                end
            end
            if control.karma_heuristic_policies
                a_bid_1_std = func.standardize_mean_var(a_bid_1, W1_bid_1, W2_bid_1);
                a_bid_1_u_std = func.standardize_mean_var(a_bid_1_u, W1_bid_1_u, W2_bid_1_u);
                a_bid_rand_std = func.standardize_mean_var(a_bid_rand, W1_bid_rand, W2_bid_rand);
                a_bid_rand_u_std = func.standardize_mean_var(a_bid_rand, W1_bid_rand_u, W2_bid_rand_u);
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
                a_lim_mem_u_std = cell(param.num_lim_mem_steps, 1);
                for i_lim_mem = 1 : param.num_lim_mem_steps
                    a_lim_mem_u_std{i_lim_mem} = func.order_rank(a_lim_mem_u{i_lim_mem});
                end
            end
            if control.karma_heuristic_policies
                a_bid_1_std = func.order_rank(a_bid_1);
                a_bid_1_u_std = func.order_rank(a_bid_1_u);
                a_bid_rand_std = func.order_rank(a_bid_rand);
                a_bid_rand_u_std = func.order_rank(a_bid_rand);
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
                a_lim_mem_u_std = cell(param.num_lim_mem_steps, 1);
                for i_lim_mem = 1 : param.num_lim_mem_steps
                    a_lim_mem_u_std{i_lim_mem} = func.order_rank_norm(a_lim_mem_u{i_lim_mem});
                end
            end
            if control.karma_heuristic_policies
                a_bid_1_std = func.order_rank_norm(a_bid_1);
                a_bid_1_u_std = func.order_rank_norm(a_bid_1_u);
                a_bid_rand_std = func.order_rank_norm(a_bid_rand);
                a_bid_rand_u_std = func.order_rank_norm(a_bid_rand);
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
end

%% Autocorrelation of accumulated cost
% Provides indication on how well population cost 'mixes' with time
if control.compute_a_acorr
    fprintf('Computing autocorrelation for baseline-random\n');
    [a_rand_acorr, acorr_tau] = func.autocorrelation(a_rand_std);
    fprintf('Computing autocorrelation for centralized-urgency\n');
    a_1_acorr = func.autocorrelation(a_1_std);
    fprintf('Computing autocorrelation for centralized-cost\n');
    a_2_acorr = func.autocorrelation(a_2_std);
    fprintf('Computing autocorrelation for centralized-urgency-then-cost\n');
    a_1_2_acorr = func.autocorrelation(a_1_2_std);
    if control.lim_mem_policies
        a_lim_mem_u_acorr = cell(param.num_lim_mem_steps, 1);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            fprintf('Computing autocorrelation for centralized-urgency-then-cost-mem-%d\n', param.lim_mem_steps(i_lim_mem));
            a_lim_mem_u_acorr{i_lim_mem} = func.autocorrelation(a_lim_mem_u_std{i_lim_mem});
        end
    end
    if control.karma_heuristic_policies
        fprintf('Computing autocorrelation for bid-1-always\n');
        a_bid_1_acorr = func.autocorrelation(a_bid_1_std);
        fprintf('Computing autocorrelation for bid-1-if-urgent\n');
        a_bid_1_u_acorr = func.autocorrelation(a_bid_1_u_std);
        fprintf('Computing autocorrelation for bid-random-always\n');
        a_bid_rand_acorr = func.autocorrelation(a_bid_rand_std);
        fprintf('Computing autocorrelation for bid-random-if-urgent\n');
        a_bid_rand_u_acorr = func.autocorrelation(a_bid_rand_u_std);
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

%% Store results
if param.save
    fprintf('Saving workspace\n');
    if control.karma_heuristic_policies || control.karma_ne_policies || control.karma_sw_policy
        save(['results/k_max_', num2str(param.k_max, '%02d'), '_k_ave_', num2str(param.k_ave, '%02d'), '.mat']);
    else
        save('results/centralized_policies.mat');
    end
end

%% Plots
if param.plot
    fprintf('Plotting\n');
    do_plots;
end

%% Inform user when done
fprintf('DONE\n\n');