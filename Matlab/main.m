clear;
close all;
clc;

%% Control randomization
rng(0);
[u_stream, agent_stream, karma_stream] = RandStream.create('mrg32k3a', 'NumStreams', 3);

%% Code control bits
% Flag to simulate centralized limited memory policies
control.lim_mem_policies = false;

% Flag to simulate heuristic karma policies
control.karma_heuristic_policies = false;

% Flag to simulate karma Nash equilibrium policies
control.karma_ne_policies = true;

% Flag to simulate karma social welfare policy
control.karma_sw_policy = true;

% Flag to compute entropy of limited memory policies & karma
control.compute_entropy = false;

% Autocorrelation takes long time to compute
control.compute_a_acorr = false;

%% Parameters
param = load_parameters();

%% Simulation initialization
% Populatin of agent indices to sample from
population = 1 : param.N;

%% Cost matrices for different policies
% Row => Time step
% Col => Agent

% Benchmark policies
% Cost for baseline policy - random
c_rand = func.allocate_cost(param);
% Cost for centralized urgency
c_u = func.allocate_cost(param);
% Cost for centralized service ratio
c_sr = func.allocate_cost(param);
% Cost for centralized urgency then service ratio
c_u_sr = func.allocate_cost(param);

% Centralized policies with limited memory
if control.lim_mem_policies
    c_lim_mem_sr = cell(param.num_lim_mem_steps, 1);
    c_lim_mem_u_sr = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        c_lim_mem_sr{i_lim_mem} = func.allocate_cost(param);
        c_lim_mem_u_sr{i_lim_mem} = func.allocate_cost(param);
    end
end

% Hueristic karma policies
if control.karma_heuristic_policies
    % Cost for bid 1 always policy
    c_bid_1 = func.allocate_cost(param);
    % Cost for bid urgency policy
    c_bid_u = func.allocate_cost(param);
    % Cost for bid all always policy
    c_bid_all = func.allocate_cost(param);
    % Cost for bid all if urgent policy
    c_bid_all_u = func.allocate_cost(param);
    % Cost for bid random always policy
    c_bid_rand = func.allocate_cost(param);
    % Cost for bid random if urgent policy
    c_bid_rand_u = func.allocate_cost(param);
end

% Nash equilibrium karma policies
if control.karma_ne_policies
    c_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        c_ne{i_alpha} = func.allocate_cost(param);
    end
end

% Social welfare karma policy
if control.karma_sw_policy
    c_sw = func.allocate_cost(param);
end

%% Karma matrices for different karma policies
% Initial karma, for cases when it is common amongst multiple policies
if control.karma_heuristic_policies || (control.karma_ne_policies || control.karma_sw_policy) && param.karma_initialization ~= 2
    if param.karma_initialization == 0
        init_k = param.k_ave * ones(1, param.N);
    else
        [s_up_k_uniform, K_uniform] = func.get_s_up_k_uniform(param.k_ave);
        init_k = func.get_init_k(s_up_k_uniform, K_uniform, param);
    end
end
    
% Heuristic karma policies
if control.karma_heuristic_policies
    % Karma for bid 1 always policy
    k_bid_1 = func.allocate_karma(param, init_k);
    % Karma for bid urgency policy
    k_bid_u = func.allocate_karma(param, init_k);
    % Karma for bid all always policy
    k_bid_all = func.allocate_karma(param, init_k);
    % Karma for bid all if urgent policy
    k_bid_all_u = func.allocate_karma(param, init_k);
    % Karma for bid random always policy
    k_bid_rand = func.allocate_karma(param, init_k);
    % Karma for bid random if urgent policy
    k_bid_rand_u = func.allocate_karma(param, init_k);
end

% Nash equilibrium karma policies
if control.karma_ne_policies
    K_ne = cell(param.num_alpha, 1);
    k_ne = cell(param.num_alpha, 1);
    ne_file_str = 'karma_nash_equilibrium/results/ne_U_';
    for i_u = 1 : param.num_U
        ne_file_str = [ne_file_str, num2str(param.U(i_u)), '_'];
    end
    ne_file_str = [ne_file_str, 'p_'];
    if isnan(param.mu_bias)
        for i_u = 1 : param.num_U
            for i_un = 1 : param.num_U
                ne_file_str = [ne_file_str, num2str(param.mu_down_u_up_un(i_u,i_un), '%.2f'), '_'];
            end
        end
    else
        ne_file_str = [ne_file_str, num2str(param.mu_bias, '%.2f'), '_'];
    end
    ne_file_str = [ne_file_str, 'm_', num2str(param.m_exchange),...
        '/k_ave_', num2str(param.k_ave, '%02d'),...
        '_alpha_'];
    for i_alpha = 1 : param.num_alpha
        alpha = param.alpha(i_alpha);
        if alpha > 0.99 && alpha < 1
            ne_file = [ne_file_str, num2str(alpha, '%.3f'), '.mat'];
        else
            ne_file = [ne_file_str, num2str(alpha, '%.2f'), '.mat'];
        end
        load(ne_file, 'ne_param');
        K_ne{i_alpha} = ne_param.K;
        
        if param.karma_initialization == 2
            % Initialize karma as per stationary distribution predicted by NE
            % algorithm
            load(ne_file, 'ne_s_up_k');
            ne_init_k = func.get_init_k(ne_s_up_k, K_ne{i_alpha}, param);
        else
            ne_init_k = init_k;
        end
        k_ne{i_alpha} = func.allocate_karma(param, ne_init_k);
    end
end

% Social welfare karma policy
if control.karma_sw_policy
    sw_file = 'karma_nash_equilibrium/results/sw_U_';
    for i_u = 1 : param.num_U
        sw_file = [sw_file, num2str(param.U(i_u)), '_'];
    end
    sw_file = [sw_file, 'p_'];
    if isnan(param.mu_bias)
        for i_u = 1 : param.num_U
            for i_un = 1 : param.num_U
                sw_file = [sw_file, num2str(param.mu_down_u_up_un(i_u,i_un), '%.2f'), '_'];
            end
        end
    else
        sw_file = [sw_file, num2str(param.mu_bias, '%.2f'), '_'];
    end
    sw_file = [sw_file, 'm_', num2str(param.m_exchange),...
        '/k_ave_', num2str(param.k_ave, '%02d'), '.mat'];
    load(sw_file, 'ne_param');
    K_sw = ne_param.K;
    
    if param.karma_initialization == 2
        % Initialize karma as per stationary distribution predicted by SW
        % algorithm
        load(sw_file, 'sw_s_up_k');
        sw_init_k = func.get_init_k(sw_s_up_k, K_sw, param);
    else
        sw_init_k = init_k;
    end
    k_sw = func.allocate_karma(param, sw_init_k);
end

%% Policy matrices for different karma policies
% Nash equilibrium karma policies
if control.karma_ne_policies
    pi_ne = cell(param.num_alpha, 1);
    pi_ne_pure = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        alpha = param.alpha(i_alpha);
        if alpha > 0.99 && alpha < 1
            ne_file = [ne_file_str, num2str(alpha, '%.3f'), '.mat'];
        else
            ne_file = [ne_file_str, num2str(alpha, '%.2f'), '.mat'];
        end
        load(ne_file, 'ne_pi_down_u_k_up_m');
        pi_ne{i_alpha} = ne_pi_down_u_k_up_m;
        pi_ne_pure{i_alpha} = func.get_pure_policy(pi_ne{i_alpha}, K_ne{i_alpha}, param);
    end
end

% Social welfare karma policy
if control.karma_sw_policy
    load(sw_file, 'sw_pi_down_u_k_up_m');
    pi_sw = sw_pi_down_u_k_up_m;
    pi_sw_pure = func.get_pure_policy(pi_sw, K_sw, param);
end

%% Urgency initialization
u_today = zeros(1, param.N);
u_hist = func.allocate_cost(param);
if param.mu_bias ~= 0.5
    % Initialize urgency for N agents as per staionary urgency distribution
    u_last = zeros(1, param.N);
    start_i = 0;
    for i_u = 1 : param.num_U
        num_agents = round(param.p_up_u(i_u) * param.N);
        u_last(start_i+1:start_i+num_agents) = param.U(i_u);
        start_i = start_i + num_agents;
    end
    u_last = u_last(randperm(u_stream, param.N));
end


%% Number of times each agent was in an interaction
num_inter = zeros(1, param.N);

%% Simulation run
% Convention:   win_id := id of agent(s) that win
%               lose_id := id of agent(s) that lose
for day = 1 : param.num_days
    % At the end of the warm-up period, reset everything
    if day == param.warm_up_days
        % Reset cost matrices
        c_rand = func.allocate_cost(param);
        c_u = func.allocate_cost(param);
        c_sr = func.allocate_cost(param);
        c_u_sr = func.allocate_cost(param);
        if control.lim_mem_policies
            for i_lim_mem = 1 : param.num_lim_mem_steps
                c_lim_mem_sr{i_lim_mem} = func.allocate_cost(param);
                c_lim_mem_u_sr{i_lim_mem} = func.allocate_cost(param);
            end
        end
        if control.karma_heuristic_policies
            c_bid_1 = func.allocate_cost(param);
            c_bid_u = func.allocate_cost(param);
            c_bid_all = func.allocate_cost(param);
            c_bid_all_u = func.allocate_cost(param);
            c_bid_rand = func.allocate_cost(param);
            c_bid_rand_u = func.allocate_cost(param);
        end
        if control.karma_ne_policies
            for i_alpha = 1 : param.num_alpha
                c_ne{i_alpha} = func.allocate_cost(param);
            end
        end
        if control.karma_sw_policy
            c_sw = func.allocate_cost(param);
        end

        % Reset karma matrices to the karma distribution at the end of
        % the warm-up period
        if control.karma_heuristic_policies
            warm_up_k = func.get_end_karma(k_bid_1, num_inter, param);
            k_bid_1 = func.allocate_karma(param, warm_up_k);
            warm_up_k = func.get_end_karma(k_bid_u, num_inter, param);
            k_bid_u = func.allocate_karma(param, warm_up_k);
            warm_up_k = func.get_end_karma(k_bid_all, num_inter, param);
            k_bid_all = func.allocate_karma(param, warm_up_k);
            warm_up_k = func.get_end_karma(k_bid_all_u, num_inter, param);
            k_bid_all_u = func.allocate_karma(param, warm_up_k);
            warm_up_k = func.get_end_karma(k_bid_rand, num_inter, param);
            k_bid_rand = func.allocate_karma(param, warm_up_k);
            warm_up_k = func.get_end_karma(k_bid_rand_u, num_inter, param);
            k_bid_rand_u = func.allocate_karma(param, warm_up_k);
        end
        if control.karma_ne_policies
            for i_alpha = 1 : param.num_alpha
                warm_up_k = func.get_end_karma(k_ne{i_alpha}, num_inter, param);
                k_ne{i_alpha} = func.allocate_karma(param, warm_up_k);
            end
        end
        if control.karma_sw_policy
            warm_up_k = func.get_end_karma(k_sw, num_inter, param);
            k_sw = func.allocate_karma(param, warm_up_k);
        end

        % Reset number of interactions
        num_inter = zeros(1, param.N);
    end
    
    if param.num_inter_per_day > 1
        % Urgency stays constant for agents per day
        if param.mu_bias == 0.5
            u_today = datasample(u_stream, param.U, param.N).';
        else
            % Sample from respective transition probabilities based on last
            % u
            for i_u_last = 1 : param.num_U
                i_agents_u_last = find(u_last == param.U(i_u_last));
                u_today(i_agents_u_last) = datasample(u_stream, param.U, length(i_agents_u_last), 'Weights', param.mu_down_u_up_un(i_u_last,:)).';
            end
            u_last = u_today;
        end
    end
    
    for inter = 1 : param.num_inter_per_day
        t = (day - 1) * param.num_inter_per_day + inter;
        % Tell user where we are
        fprintf('Day: %d Interaction: %d Timestep: %d\n', day, inter, t);
        
        if ~param.same_num_inter
            % Sample agents i & j uniformly at random
            agents_id = datasample(agent_stream, population, param.I_size, 'Replace', false);
        else
            % If all population has been sampled, re-fill population
            if isempty(population)
                population = 1 : param.N;
            end
            % Sample agents i & j uniformly at random and remove them from
            % population
            agents_id = datasample(agent_stream, population, param.I_size, 'Replace', false);
            for i_agent = 1 : param.I_size
                population(population == agents_id(i_agent)) = [];
            end
        end

        % Increment number of interactions for picked agents
        num_inter(agents_id) = num_inter(agents_id) + 1;
        
        % Assert if an agent is picked too many times for memory allocated
        assert(max(num_inter(agents_id)) < param.max_num_inter_per_agent,...
            'One agent was picked too many times. Increase maximum allowed number of interactions per agent.');

        % Urgency of sampled agents
        if param.num_inter_per_day > 1
            u = u_today(agents_id);
        else
            if param.mu_bias == 0.5
                u = datasample(u_stream, param.U, param.I_size).';
            else
                u = zeros(1, param.I_size);
                for i_agent = 1 : param.I_size
                    i_u_last = find(param.U == u_last(agents_id(i_agent)));
                    u(i_agent) = datasample(u_stream, param.U, 1, 'Weights', param.mu_down_u_up_un(i_u_last,:));
                end
                u_last(agents_id) = u;
            end
        end
        
        % Update history of urgency
        for i_agent = 1 : param.I_size
            id = agents_id(i_agent);
            u_hist(num_inter(id),id) = u(i_agent);
        end
        

        %% RANDOM POLICY %%
        % Simply choose 'first' agent, since there is already randomization
        % in the agent order
        win_id = agents_id(1);

        % Update cost. Winning agent incurs 0 cost. Losing agent incurs
        % cost equal to their urgency
        for i_agent = 1 : param.I_size
            id = agents_id(i_agent);
            if id == win_id
                c_rand(num_inter(id),id) = 0;
            else
                c_rand(num_inter(id),id) = u(i_agent);
            end
        end
        
        %% CENTRALIZED POLICIES %%
        %% Centralized urgency policy
        % Agent with max urgency passes. Coin flip on tie
        % Find agent(s) with max urgency, which are candidates for passing
        [~, i_win] = func.multi_maxes(u);
        win_max_u = agents_id(i_win);
        num_max_u = length(win_max_u);
        
        % Choose 'first' urgent agent (there is already randomization in
        % the agent order)
        win_id = win_max_u(1);

        % Update cost. Winning agent incurs 0 cost. Losing agent incurs
        % cost equal to their urgency
        for i_agent = 1 : param.I_size
            id = agents_id(i_agent);
            if id == win_id
                c_u(num_inter(id),id) = 0;
            else
                c_u(num_inter(id),id) = u(i_agent);
            end
        end

        %% Centralized service ratio policy
        % Agent with maximum relative cost (counting current urgency)
        % passes. Coin flip on tie
        r = func.relative_cost(c_sr, agents_id, u_hist, num_inter, inf);
        
        [~, i_win] = max(r);
        win_id = agents_id(i_win);

        % Update cost. Winning agent incurs 0 cost. Losing agent incurs
        % cost equal to their urgency
        for i_agent = 1 : param.I_size
            id = agents_id(i_agent);
            if id == win_id
                c_sr(num_inter(id),id) = 0;
            else
                c_sr(num_inter(id),id) = u(i_agent);
            end
        end

        %% Centralized urgency then service ratio policy
        % Agent(s) with max urgency, which are candidates for passing, were
        % already found in first step of centralized urgency policy
        % If there are multiple agents with max urgency, pick one based on
        % relative cost like in centralized relative cost policy
        if num_max_u > 1
            r = func.relative_cost(c_u_sr, win_max_u, u_hist, num_inter, inf);
            [~, i_win] = max(r);
            win_id = win_max_u(i_win);
        else
            win_id = win_max_u;
        end

        % Update cost. Winning agent incurs 0 cost. Losing agent incurs
        % cost equal to their urgency
        for i_agent = 1 : param.I_size
            id = agents_id(i_agent);
            if id == win_id
                c_u_sr(num_inter(id),id) = 0;
            else
                c_u_sr(num_inter(id),id) = u(i_agent);
            end
        end

        %% Centralized policies with limited memroy
        if control.lim_mem_policies
            for i_lim_mem = 1 : param.num_lim_mem_steps
                %% Centralized accumulated cost with limited memory
                % Agent with maximum accumulated cost up to limited number
                % of interactions (counting current urgency) passes
                % Coin flip on tie
                r = func.relative_cost(c_sr, agents_id, u_hist, num_inter, param.lim_mem_steps(i_lim_mem));
                [~, i_win] = max(r);
                win_id = agents_id(i_win);

                % Update cost. Winning agent incurs 0 cost. Losing agent
                % incurs cost equal to their urgency
                for i_agent = 1 : param.I_size
                    id = agents_id(i_agent);
                    if id == win_id
                        c_lim_mem_sr{i_lim_mem}(num_inter(id),id) = 0;
                    else
                        c_lim_mem_sr{i_lim_mem}(num_inter(id),id) = u(i_agent);
                    end
                end
                
                %% Centralized urgency then accumulated cost with limited memory
                % Agent(s) with max urgency, which are candidates for passing, were
                % already found in first step of centralized policy 1
                % If there are multiple agents with max urgency, pick on based on
                % accumulated cost up to limited number of interactions per
                % agent
                if num_max_u > 1
                    r = func.relative_cost(c_u_sr, win_max_u, u_hist, num_inter, param.lim_mem_steps(i_lim_mem));
                    [~, i_win] = max(r);
                    win_id = win_max_u(i_win);
                else
                    win_id = win_max_u;
                end

                % Update cost. Winning agent incurs 0 cost. Losing agent
                % incurs cost equal to their urgency
                for i_agent = 1 : param.I_size
                    id = agents_id(i_agent);
                    if id == win_id
                        c_lim_mem_u_sr{i_lim_mem}(num_inter(id),id) = 0;
                    else
                        c_lim_mem_u_sr{i_lim_mem}(num_inter(id),id) = u(i_agent);
                    end
                end
            end
        end
        
        %% KARMA POLICIES %%
        % Save state of random stream to revert to before randomization of
        % the different policies. This ensures 'ties' are handled the same
        karma_stream_state = karma_stream.State;
        
        %% Heuristic karma policies
        if control.karma_heuristic_policies
            %% Bid 1 always policy
            % Agents simply bid 1, if they have it
            m = zeros(1, param.I_size);
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                m(i_agent) = min([1, k_bid_1(num_inter(id),id)]);
            end

            % Agent bidding max karma passes
            [m_win, i_win] = max(m);
            win_id = agents_id(i_win);

            % Update cost. Winning agent incurs 0 cost. Losing agent incurs
            % cost equal to their urgency
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    c_bid_1(num_inter(id),id) = 0;
                else
                    c_bid_1(num_inter(id),id) = u(i_agent);
                end
            end

            % Get losing bid
            m_lose = min(m);

            % Get karma payment
            p = func.get_karma_payment(m_win, m_lose, param);
            
            % Update karma based on payment and winning agent
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    k_bid_1(num_inter(id)+1,id) = k_bid_1(num_inter(id),id) - p;
                else
                    k_bid_1(num_inter(id)+1,id) = k_bid_1(num_inter(id),id) + p;
                end
            end

            %% Bid urgency policy
            % Agents bid their level of urgency, starting with 0 for lowest
            % level, if they have it
            m = zeros(1, param.I_size);
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                i_u = find(param.U == u(i_agent));
                m(i_agent) = min(i_u - 1, k_bid_u(num_inter(id),id));
            end

            % Agent bidding max karma passes
            [m_win, i_win] = max(m);
            win_id = agents_id(i_win);

            % Update cost. Winning agent incurs 0 cost. Losing agent incurs
            % cost equal to their urgency
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    c_bid_u(num_inter(id),id) = 0;
                else
                    c_bid_u(num_inter(id),id) = u(i_agent);
                end
            end

            % Get losing bid
            m_lose = min(m);

            % Get karma payment
            p = func.get_karma_payment(m_win, m_lose, param);
            
            % Update karma based on payment and winning agent
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    k_bid_u(num_inter(id)+1,id) = k_bid_u(num_inter(id),id) - p;
                else
                    k_bid_u(num_inter(id)+1,id) = k_bid_u(num_inter(id),id) + p;
                end
            end

            %% Bid all always policy
            % Agents bid all their karma
            m = zeros(1, param.I_size);
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                m(i_agent) = k_bid_all(num_inter(id),id);
            end

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = max(m);
            win_id = agents_id(i_win);

            % Update cost. Winning agent incurs 0 cost. Losing agent incurs
            % cost equal to their urgency
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    c_bid_all(num_inter(id),id) = 0;
                else
                    c_bid_all(num_inter(id),id) = u(i_agent);
                end
            end

            % Get losing bid
            m_lose = min(m);

            % Get karma payment
            p = func.get_karma_payment(m_win, m_lose, param);
            
            % Update karma based on payment and winning agent
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    k_bid_all(num_inter(id)+1,id) = k_bid_all(num_inter(id),id) - p;
                else
                    k_bid_all(num_inter(id)+1,id) = k_bid_all(num_inter(id),id) + p;
                end
            end

            %% Bid all if urgent policy
            % Agents bid all their karma, if they are urgent
            m = zeros(1, param.I_size);
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                m(i_agent) = k_bid_all_u(num_inter(id),id);
            end
            m(u == param.u_min) = 0;

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = max(m);
            win_id = agents_id(i_win);

            % Update cost. Winning agent incurs 0 cost. Losing agent incurs
            % cost equal to their urgency
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    c_bid_all_u(num_inter(id),id) = 0;
                else
                    c_bid_all_u(num_inter(id),id) = u(i_agent);
                end
            end

            % Get losing bid
            m_lose = min(m);

            % Get karma payment
            p = func.get_karma_payment(m_win, m_lose, param);
            
            % Update karma based on payment and winning agent
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    k_bid_all_u(num_inter(id)+1,id) = k_bid_all_u(num_inter(id),id) - p;
                else
                    k_bid_all_u(num_inter(id)+1,id) = k_bid_all_u(num_inter(id),id) + p;
                end
            end
            
            %% Bid random always policy
            % Agents simply bid random
            m = zeros(1, param.I_size);
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                m(i_agent) = datasample(karma_stream, 0 : k_bid_rand(num_inter(id),id), 1);
            end

            % Agent bidding max karma passes
            [m_win, i_win] = max(m);
            win_id = agents_id(i_win);

            % Update cost. Winning agent incurs 0 cost. Losing agent incurs
            % cost equal to their urgency
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    c_bid_rand(num_inter(id),id) = 0;
                else
                    c_bid_rand(num_inter(id),id) = u(i_agent);
                end
            end

            % Get losing bid
            m_lose = min(m);

            % Get karma payment
            p = func.get_karma_payment(m_win, m_lose, param);
            
            % Update karma based on payment and winning agent
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    k_bid_rand(num_inter(id)+1,id) = k_bid_rand(num_inter(id),id) - p;
                else
                    k_bid_rand(num_inter(id)+1,id) = k_bid_rand(num_inter(id),id) + p;
                end
            end

            %% Bid random if urgent policy
            % Agents bid random if urgent, greater than zero if they can
            karma_stream.State = karma_stream_state;
            m = zeros(1, param.I_size);
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                k = k_bid_rand_u(num_inter(id),id);
                if u(i_agent) ~= param.u_min && k > 0
                    m(i_agent) = datasample(karma_stream, 1 : k, 1);
                end
            end

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = max(m);
            win_id = agents_id(i_win);

            % Update cost. Winning agent incurs 0 cost. Losing agent incurs
            % cost equal to their urgency
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    c_bid_rand_u(num_inter(id),id) = 0;
                else
                    c_bid_rand_u(num_inter(id),id) = u(i_agent);
                end
            end

            % Get losing bid
            m_lose = min(m);

            % Get karma payment
            p = func.get_karma_payment(m_win, m_lose, param);
            
            % Update karma based on payment and winning agent
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    k_bid_rand_u(num_inter(id)+1,id) = k_bid_rand_u(num_inter(id),id) - p;
                else
                    k_bid_rand_u(num_inter(id)+1,id) = k_bid_rand_u(num_inter(id),id) + p;
                end
            end
        end
        
        %% Nash equilibrium karma policies
        if control.karma_ne_policies
            for i_alpha = 1 : param.num_alpha
                % Get agents' bids from their policies
                karma_stream.State = karma_stream_state;
                m = zeros(1, param.I_size);
                for i_agent = 1 : param.I_size
                    id = agents_id(i_agent);
                    k = k_ne{i_alpha}(num_inter(id),id);
                    i_u = find(param.U == u(i_agent));
                    i_k = min([k + 1, length(K_ne{i_alpha})]); % Uses policy of k_max if it is exceeded
                    m(i_agent) = pi_ne_pure{i_alpha}(i_u,i_k);
                    if isnan(m(i_agent))
                        m(i_agent) = datasample(karma_stream, K_ne{i_alpha}, 1, 'Weights', squeeze(pi_ne{i_alpha}(i_u,i_k,:)));
                    end
                end
                
                % Agent bidding max karma passes and pays karma bidded
                [m_win, i_win] = max(m);
                win_id = agents_id(i_win);

                % Update cost. Winning agent incurs 0 cost. Losing agent
                % incurs cost equal to their urgency
                for i_agent = 1 : param.I_size
                    id = agents_id(i_agent);
                    if id == win_id
                        c_ne{i_alpha}(num_inter(id),id) = 0;
                    else
                        c_ne{i_alpha}(num_inter(id),id) = u(i_agent);
                    end
                end

                % Get losing bid
                m_lose = min(m);

                % Get karma payment
                p = func.get_karma_payment(m_win, m_lose, param);

                % Update karma based on payment and winning agent
                for i_agent = 1 : param.I_size
                    id = agents_id(i_agent);
                    if id == win_id
                        k_ne{i_alpha}(num_inter(id)+1,id) = k_ne{i_alpha}(num_inter(id),id) - p;
                    else
                        k_ne{i_alpha}(num_inter(id)+1,id) = k_ne{i_alpha}(num_inter(id),id) + p;
                    end
                end
            end
        end
        
        %% Social welfare karma policy
        if control.karma_sw_policy
            % Get agents' bids from their policies
            karma_stream.State = karma_stream_state;
            m = zeros(1, param.I_size);
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                k = k_sw(num_inter(id),id);
                i_u = find(param.U == u(i_agent));
                i_k = min([k + 1, length(K_sw)]); % Uses policy of k_max if it is exceeded
                m(i_agent) = pi_sw_pure(i_u,i_k);
                if isnan(m(i_agent))
                    m(i_agent) = datasample(karma_stream, K_sw, 1, 'Weights', squeeze(pi_sw(i_u,i_k,:)));
                end
            end

            % Agent bidding max karma passes and pays karma bidded
            [m_win, i_win] = max(m);
            win_id = agents_id(i_win);

            % Update cost. Winning agent incurs 0 cost. Losing agent incurs
            % cost equal to their urgency
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    c_sw(num_inter(id),id) = 0;
                else
                    c_sw(num_inter(id),id) = u(i_agent);
                end
            end

            % Get losing bid
            m_lose = min(m);

            % Get karma payment
            p = func.get_karma_payment(m_win, m_lose, param);
            
            % Update karma based on payment and winning agent
            for i_agent = 1 : param.I_size
                id = agents_id(i_agent);
                if id == win_id
                    k_sw(num_inter(id)+1,id) = k_sw(num_inter(id),id) - p;
                else
                    k_sw(num_inter(id)+1,id) = k_sw(num_inter(id),id) + p;
                end
            end
        end
    end
end

%% Perfromance measures
fprintf('Computing performance measures\n');

%% Accumulated costs per agent at each time step
fprintf('Computing accumulated costs\n');
a_rand = func.get_accumulated_cost(c_rand, num_inter, param);
a_u = func.get_accumulated_cost(c_u, num_inter, param);
a_sr = func.get_accumulated_cost(c_sr, num_inter, param);
a_u_sr = func.get_accumulated_cost(c_u_sr, num_inter, param);
if control.lim_mem_policies
    a_lim_mem_sr = cell(param.num_lim_mem_steps, 1);
    a_lim_mem_u_sr = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        a_lim_mem_sr{i_lim_mem} = func.get_accumulated_cost(c_lim_mem_sr{i_lim_mem}, num_inter, param);
        a_lim_mem_u_sr{i_lim_mem} = func.get_accumulated_cost(c_lim_mem_u_sr{i_lim_mem}, num_inter, param);
    end
end
if control.karma_heuristic_policies
    a_bid_1 = func.get_accumulated_cost(c_bid_1, num_inter, param);
    a_bid_u = func.get_accumulated_cost(c_bid_u, num_inter, param);
    a_bid_all = func.get_accumulated_cost(c_bid_all, num_inter, param);
    a_bid_all_u = func.get_accumulated_cost(c_bid_all_u, num_inter, param);
    a_bid_rand = func.get_accumulated_cost(c_bid_rand, num_inter, param);
    a_bid_rand_u = func.get_accumulated_cost(c_bid_rand_u, num_inter, param);
end
if control.karma_ne_policies
    a_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        a_ne{i_alpha} = func.get_accumulated_cost(c_ne{i_alpha}, num_inter, param);
    end
end
if control.karma_sw_policy
    a_sw = func.get_accumulated_cost(c_sw, num_inter, param);
end

%% Relative costs per agent at each time step
fprintf('Computing relative costs\n');
r_rand = func.get_relative_cost(c_rand, u_hist, num_inter, param);
r_u = func.get_relative_cost(c_u, u_hist, num_inter, param);
r_a = func.get_relative_cost(c_sr, u_hist, num_inter, param);
r_u_a = func.get_relative_cost(c_u_sr, u_hist, num_inter, param);
if control.lim_mem_policies
    r_lim_mem_a = cell(param.num_lim_mem_steps, 1);
    r_lim_mem_u_a = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        r_lim_mem_a{i_lim_mem} = func.get_relative_cost(c_lim_mem_sr{i_lim_mem}, u_hist, num_inter, param);
        r_lim_mem_u_a{i_lim_mem} = func.get_relative_cost(c_lim_mem_u_sr{i_lim_mem}, u_hist, num_inter, param);
    end
end
if control.karma_heuristic_policies
    r_bid_1 = func.get_relative_cost(c_bid_1, u_hist, num_inter, param);
    r_bid_u = func.get_relative_cost(c_bid_u, u_hist, num_inter, param);
    r_bid_all = func.get_relative_cost(c_bid_all, u_hist, num_inter, param);
    r_bid_all_u = func.get_relative_cost(c_bid_all_u, u_hist, num_inter, param);
    r_bid_rand = func.get_relative_cost(c_bid_rand, u_hist, num_inter, param);
    r_bid_rand_u = func.get_relative_cost(c_bid_rand_u, u_hist, num_inter, param);
end
if control.karma_ne_policies
    r_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        r_ne{i_alpha} = func.get_relative_cost(c_ne{i_alpha}, u_hist, num_inter, param);
    end
end
if control.karma_sw_policy
    r_sw = func.get_relative_cost(c_sw, u_hist, num_inter, param);
end

%% Inefficiency vs. time
fprintf('Computing efficiencies\n');
IE_rand = mean(a_rand, 2);
IE_u = mean(a_u, 2);
IE_a = mean(a_sr, 2);
IE_u_a = mean(a_u_sr, 2);
if control.lim_mem_policies
    IE_lim_mem_a = cell(param.num_lim_mem_steps, 1);
    IE_lim_mem_u_a = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        IE_lim_mem_a{i_lim_mem} = mean(a_lim_mem_sr{i_lim_mem}, 2);
        IE_lim_mem_u_a{i_lim_mem} = mean(a_lim_mem_u_sr{i_lim_mem}, 2);
    end
end
if control.karma_heuristic_policies
    IE_bid_1 = mean(a_bid_1, 2);
    IE_bid_u = mean(a_bid_u, 2);
    IE_bid_all = mean(a_bid_all, 2);
    IE_bid_all_u = mean(a_bid_all_u, 2);
    IE_bid_rand = mean(a_bid_rand, 2);
    IE_bid_rand_u = mean(a_bid_rand_u, 2);
end
if control.karma_ne_policies
    IE_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        IE_ne{i_alpha} = mean(a_ne{i_alpha}, 2);
    end
end
if control.karma_sw_policy
    IE_sw = mean(a_sw, 2);
end

%% Unfairness vs. time
fprintf('Computing fairness\n');
UF_rand = var(r_rand, [], 2);
UF_u = var(r_u, [], 2);
UF_a = var(r_a, [], 2);
UF_u_a = var(r_u_a, [], 2);
if control.lim_mem_policies
    UF_lim_mem_a = cell(param.num_lim_mem_steps, 1);
    UF_lim_mem_u_a = cell(param.num_lim_mem_steps, 1);
    for i_lim_mem = 1 : param.num_lim_mem_steps
        UF_lim_mem_a{i_lim_mem} = var(r_lim_mem_a{i_lim_mem}, [], 2);
        UF_lim_mem_u_a{i_lim_mem} = var(r_lim_mem_u_a{i_lim_mem}, [], 2);
    end
end
if control.karma_heuristic_policies
    UF_bid_1 = var(r_bid_1, [], 2);
    UF_bid_u = var(r_bid_u, [], 2);
    UF_bid_all = var(r_bid_all, [], 2);
    UF_bid_all_u = var(r_bid_all_u, [], 2);
    UF_bid_rand = var(r_bid_rand, [], 2);
    UF_bid_rand_u = var(r_bid_rand_u, [], 2);
end
if control.karma_ne_policies
    UF_ne = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        UF_ne{i_alpha} = var(r_ne{i_alpha}, [], 2);
    end
end
if control.karma_sw_policy
    UF_sw = var(r_sw, [], 2);
end

%% Karma distributions
if control.karma_heuristic_policies
    [k_bid_1_dist, k_bid_1_dist_agents] = func.get_karma_dist(k_bid_1, param);
    [k_bid_u_dist, k_bid_u_dist_agents] = func.get_karma_dist(k_bid_u, param);
    [k_bid_all_dist, k_bid_all_dist_agents] = func.get_karma_dist(k_bid_all, param);
    [k_bid_all_u_dist, k_bid_all_u_dist_agents] = func.get_karma_dist(k_bid_all_u, param);
    [k_bid_rand_dist, k_bid_rand_dist_agents] = func.get_karma_dist(k_bid_rand, param);
    [k_bid_rand_u_dist, k_bid_rand_u_dist_agents] = func.get_karma_dist(k_bid_rand_u, param);
end
if control.karma_ne_policies
    k_ne_dist = cell(param.num_alpha, 1);
    k_ne_dist_agents = cell(param.num_alpha, 1);
    for i_alpha = 1 : param.num_alpha
        [k_ne_dist{i_alpha}, k_ne_dist_agents{i_alpha}] = func.get_karma_dist(k_ne{i_alpha}, param);
    end
end
if control.karma_sw_policy
    [k_sw_dist, k_sw_dist_agents] = func.get_karma_dist(k_sw, param);
end

%% Entropy
if control.compute_entropy
    % Entropy of limited memory accumulated costs
    if control.lim_mem_policies
        ent_lim_mem_rand = zeros(param.num_lim_mem_steps, 1);
        ent_lim_mem_u = zeros(param.num_lim_mem_steps, 1);
        ent_lim_mem_a = zeros(param.num_lim_mem_steps, 1);
        ent_lim_mem_u_a = zeros(param.num_lim_mem_steps, 1);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            ent_lim_mem_rand(i_lim_mem) = func.get_entropy_lim_mem(c_rand, num_inter, param.lim_mem_steps(i_lim_mem), param);
            ent_lim_mem_u(i_lim_mem) = func.get_entropy_lim_mem(c_u, num_inter, param.lim_mem_steps(i_lim_mem), param);
            ent_lim_mem_a(i_lim_mem) = func.get_entropy_lim_mem(c_lim_mem_sr{i_lim_mem}, num_inter, param.lim_mem_steps(i_lim_mem), param);
            ent_lim_mem_u_a(i_lim_mem) = func.get_entropy_lim_mem(c_lim_mem_u_sr{i_lim_mem}, num_inter, param.lim_mem_steps(i_lim_mem), param);
        end
    end
    
    % Entropy of karma
    if control.karma_heuristic_policies
        ent_bid_1 = func.get_entropy(k_bid_1_dist);
        ent_bid_u = func.get_entropy(k_bid_u_dist);
        ent_bid_all = func.get_entropy(k_bid_all_dist);
        ent_bid_all_u = func.get_entropy(k_bid_all_u_dist);
        ent_bid_rand = func.get_entropy(k_bid_rand_dist);
        ent_bid_rand_u = func.get_entropy(k_bid_rand_u_dist);
    end
    if control.karma_ne_policies
        ent_ne = zeros(param.num_alpha, 1);
        for i_alpha = 1 : param.num_alpha
            ent_ne(i_alpha) = func.get_entropy(k_ne_dist{i_alpha});
        end
    end
    if control.karma_sw_policy
        ent_sw = func.get_entropy(k_sw_dist);
    end
end


%% Standardized accumulated costs
% Standardization method is a parameter. Required for autocorrelation and
% allows to investigate 'mixing' ability of policies. Only compute if
% autocorrelation flag is on
if control.compute_a_acorr
    fprintf('Standardizing accumulated costs\n');
    a_rand_std = func.get_standardized_cost(a_rand, IE_rand, UF_rand, param);
    a_u_std = func.get_standardized_cost(a_u, IE_u, UF_u, param);
    a_sr_std = func.get_standardized_cost(a_sr, IE_a, UF_a, param);
    a_u_sr_std = func.get_standardized_cost(a_u_sr, IE_u_a, UF_u_a, param);
    if control.lim_mem_policies
        a_lim_mem_sr_std = cell(param.num_lim_mem_steps, 1);
        a_lim_mem_u_sr_std = cell(param.num_lim_mem_steps, 1);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            a_lim_mem_sr_std{i_lim_mem} = func.get_standardized_cost(a_lim_mem_sr{i_lim_mem}, IE_lim_mem_a{i_lim_mem}, UF_lim_mem_a{i_lim_mem}, param);
            a_lim_mem_u_sr_std{i_lim_mem} = func.get_standardized_cost(a_lim_mem_u_sr{i_lim_mem}, IE_lim_mem_u_a{i_lim_mem}, UF_lim_mem_u_a{i_lim_mem}, param);
        end
    end
    if control.karma_heuristic_policies
        a_bid_1_std = func.get_standardized_cost(a_bid_1, IE_bid_1, UF_bid_1, param);
        a_bid_u_std = func.get_standardized_cost(a_bid_u, IE_bid_u, UF_bid_u, param);
        a_bid_all_std = func.get_standardized_cost(a_bid_all, IE_bid_all, UF_bid_all, param);
        a_bid_all_u_std = func.get_standardized_cost(a_bid_all_u, IE_bid_all_u, UF_bid_all_u, param);
        a_bid_rand_std = func.get_standardized_cost(a_bid_rand, IE_bid_rand, UF_bid_rand, param);
        a_bid_rand_u_std = func.get_standardized_cost(a_bid_rand_u, IE_bid_rand_u, UF_bid_rand_u, param);
    end
    if control.karma_ne_policies
        a_ne_std = cell(param.num_alpha, 1);
        for i_alpha = 1 : param.num_alpha
            a_ne_std{i_alpha} = func.get_standardized_cost(a_ne{i_alpha}, IE_ne{i_alpha}, UF_ne{i_alpha}, param);
        end
    end
    if control.karma_sw_policy
        a_sw_std = func.get_standardized_cost(a_sw, IE_sw, UF_sw, param);
    end
end

%% Autocorrelation of accumulated cost
% Provides indication on how well population cost 'mixes' with time
if control.compute_a_acorr
    fprintf('Computing autocorrelation for baseline-random\n');
    [a_rand_acorr, acorr_tau] = func.autocorrelation(a_rand_std);
    fprintf('Computing autocorrelation for centralized-urgency\n');
    a_u_acorr = func.autocorrelation(a_u_std);
    fprintf('Computing autocorrelation for centralized-SR\n');
    a_sr_acorr = func.autocorrelation(a_sr_std);
    fprintf('Computing autocorrelation for centralized-urgency-then-SR\n');
    a_u_sr_acorr = func.autocorrelation(a_u_sr_std);
    if control.lim_mem_policies
        a_lim_mem_sr_acorr = cell(param.num_lim_mem_steps, 1);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            fprintf('Computing autocorrelation for centralized-SR-mem-%d\n', param.lim_mem_steps(i_lim_mem));
            a_lim_mem_sr_acorr{i_lim_mem} = func.autocorrelation(a_lim_mem_sr_std{i_lim_mem});
        end
        a_lim_mem_u_sr_acorr = cell(param.num_lim_mem_steps, 1);
        for i_lim_mem = 1 : param.num_lim_mem_steps
            fprintf('Computing autocorrelation for centralized-urgency-then-SR-mem-%d\n', param.lim_mem_steps(i_lim_mem));
            a_lim_mem_u_sr_acorr{i_lim_mem} = func.autocorrelation(a_lim_mem_u_sr_std{i_lim_mem});
        end
    end
    if control.karma_heuristic_policies
        fprintf('Computing autocorrelation for bid-1-always\n');
        a_bid_1_acorr = func.autocorrelation(a_bid_1_std);
        fprintf('Computing autocorrelation for bid-urgency\n');
        a_bid_u_acorr = func.autocorrelation(a_bid_u_std);
        fprintf('Computing autocorrelation for bid-all-always\n');
        a_bid_all_acorr = func.autocorrelation(a_bid_all_std);
        fprintf('Computing autocorrelation for bid-all-if-urgent\n');
        a_bid_all_u_acorr = func.autocorrelation(a_bid_all_u_std);
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
        if param.num_alpha == 1
            save(['results/k_ave_', num2str(param.k_ave, '%02d'), '_alpha_', num2str(param.alpha, '%.2f'), '.mat']);
        else
            save(['results/k_ave_', num2str(param.k_ave, '%02d'), '.mat']);
        end
    else
        save('results/centralized_policies.mat');
    end
end

%% Plots
if param.plot
    fprintf('Plotting\n');
    do_plots;
end

%% Inform user when done and sound
fprintf('DONE\n\n');
load mtlb.mat;
sound(mtlb);