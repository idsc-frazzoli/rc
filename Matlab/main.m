clear;
close all;
clc;

%% Add functions folder to path
addpath('functions');

%% Control randomization
rng(0);
[u_stream, agent_stream, karma_stream] = RandStream.create('mrg32k3a', 'NumStreams', 3);

%% Code control bits
% Flag to simulate fairness horizon policies
control.fairness_horizon_policies = true;

% Flag to simulate karma Nash equilibrium policies
control.karma_ne_policies = true;

% Flag to simulate karma social welfare policy
control.karma_sw_policy = true;

% Flag to compute entropy of fairness policies & karma
control.compute_entropy = false;

%% Parameters
param = load_parameters();

%% Simulation initialization
% Population of agent indices to sample from
agents = 1 : param.n_a;

%% Types of the agents
agents_to_sample = agents;
agents_per_type = cell(param.n_mu, param.n_alpha);
agents_per_mu = cell(param.n_mu, 1);
n_a_per_type = zeros(param.n_mu, param.n_alpha);
n_a_per_mu = zeros(param.n_mu, 1);
agent_types = zeros(2, param.n_a);  % First row is urgency type, second is future awareness type
for i_mu = 1 : param.n_mu
    agents_per_mu{i_mu} = [];
    for i_alpha = 1 : param.n_alpha
        if i_mu == param.n_mu && i_alpha == param.n_alpha
            n_a_per_type(i_mu,i_alpha) = param.n_a - sum(n_a_per_type(:));
        else
            n_a_per_type(i_mu,i_alpha) = round(param.g_up_mu_alpha(i_mu,i_alpha) * param.n_a);
        end
        [agents_per_type{i_mu,i_alpha}, i_agents] = datasample(agent_stream, agents_to_sample, n_a_per_type(i_mu,i_alpha), 'Replace', false);
        agents_per_mu{i_mu} = [agents_per_mu{i_mu}, agents_per_type{i_mu,i_alpha}];
        agents_to_sample(i_agents) = [];
        agent_types(:,agents_per_type{i_mu,i_alpha}) = repmat([i_mu; i_alpha], 1, n_a_per_type(i_mu,i_alpha));
    end
end
n_a_per_mu = sum(n_a_per_type, 2);

%% Urgency initialization
u_hist = allocate_mat(param);
i_u_last = zeros(1, param.n_a); % Use indices for speed
for i_mu = 1 : param.n_mu
    % Initialize as per stationary distribution of Markov chain
    start_i = 0;
    for i_u = 1 : param.n_u - 1
        n_a_per_u = round(param.prob_down_mu_up_u(i_mu,i_u) * n_a_per_mu(i_mu));
        i_agents = agents_per_mu{i_mu}(start_i+1:start_i+n_a_per_u);
        i_u_last(i_agents) = i_u;
        start_i = start_i + n_a_per_u;
    end
    i_agents = agents_per_mu{i_mu}(start_i+1:end);
    i_u_last(i_agents) = param.n_u;
end

%% Cost matrices for different policies
% Row => Timestep
% Col => Agent

% Benchmark policies
% Cost for random
c_rand = allocate_mat(param);
% Cost for centralized urgency
c_u = allocate_mat(param);
% Cost for centralized accumulated cost
c_a = allocate_mat(param);
% Cost for centralized urgency then accumulated cost
c_u_a = allocate_mat(param);

% Centralized fairness horizon policies
if control.fairness_horizon_policies
    c_fair_hor_a = cell(param.n_fairness_horizon, 1);
    c_fair_hor_u_a = cell(param.n_fairness_horizon, 1);
    for i_fair_hor = 1 : param.n_fairness_horizon
        c_fair_hor_a{i_fair_hor} = allocate_mat(param);
        c_fair_hor_u_a{i_fair_hor} = allocate_mat(param);
    end
end

% Karma Nash equilibrium policies
if control.karma_ne_policies
    c_ne = cell(param.n_alpha_comp, 1);
    for i_alpha_comp = 1 : param.n_alpha_comp
        c_ne{i_alpha_comp} = allocate_mat(param);
    end
end

% Karma social welfare policy
if control.karma_sw_policy
    c_sw = allocate_mat(param);
end

%% Karma and policy matrices for different karma policies
% Initial karma, for cases when it is common amongst multiple policies
if (control.karma_ne_policies || control.karma_sw_policy) && param.karma_initialization ~= 2
    if param.karma_initialization == 0
        init_k = param.k_bar * ones(1, param.n_a);
    else
        [sigma_up_k_uniform, K_uniform] = get_sigma_up_k_uniform(param.k_bar);
        init_k = get_init_k(sigma_up_k_uniform, K_uniform, param);
    end
end

% Nash equilibrium karma policies and distributions
if control.karma_ne_policies
    K_ne = cell(param.n_alpha_comp, 1);
    pi_ne = cell(param.n_alpha_comp, 1);
    pi_ne_pure = cell(param.n_alpha_comp, 1);
    k_ne = cell(param.n_alpha_comp, 1);
    
    ne_file_str = 'karma_nash_equilibrium/results/ne_U_';
    for i_u = 1 : param.n_u
        ne_file_str = [ne_file_str, int2str(param.U(i_u)), '_'];
    end
    for i_mu = 1 : param.n_mu
        ne_file_str = [ne_file_str, 'phi', int2str(i_mu), '_'];
        for i_u = 1 : param.n_u
            for i_un = 1 : param.n_u
                ne_file_str = [ne_file_str, num2str(param.phi_down_mu_u_up_un(i_mu,i_u,i_un), '%.2f'), '_'];
            end
        end
    end
    ne_file_str = [ne_file_str, 'pay_', int2str(param.payment_rule),...
        '/k_bar_', num2str(param.k_bar, '%02d'),...
        '_alpha'];
    if param.n_alpha == 1
        for i_alpha_comp = 1 : param.n_alpha_comp
            alpha = param.Alpha(i_alpha_comp);
            % alpha = 0 yields bid all and all of the karma with one agent
            % under pay as bid rule
            if alpha == 0 && param.payment_rule == 0
                K_ne{i_alpha_comp} = (0 : param.k_tot).';
                pi_ne_pure{i_alpha_comp} = repmat(K_ne{i_alpha_comp}.', param.n_u, 1);
                k_ne{i_alpha_comp} = allocate_mat(param);
                k_ne{i_alpha_comp}(1,:) = [zeros(1, param.n_a - 1), param.k_tot];
                continue;
            end
            
            if alpha > 0.99 && alpha < 1
                ne_file = [ne_file_str, '_', num2str(alpha, '%.3f'), '.mat'];
            else
                ne_file = [ne_file_str, '_', num2str(alpha, '%.2f'), '.mat'];
            end
            
            load(ne_file, 'ne_param', 'ne_pi_down_u_k_up_b');
            K_ne{i_alpha_comp} = ne_param.K;
            pi_ne{i_alpha_comp} = ne_pi_down_u_k_up_b;
            pi_ne_pure{i_alpha_comp} = get_pure_policy(pi_ne{i_alpha_comp}, K_ne{i_alpha_comp}, param);

            if param.karma_initialization == 2
                % Initialize karma as per stationary distribution predicted by NE
                % algorithm
                load(ne_file, 'ne_sigma_up_k');
                ne_init_k = get_init_k(ne_sigma_up_k, K_ne{i_alpha_comp}, param);
            else
                ne_init_k = init_k;
            end
            k_ne{i_alpha_comp} = allocate_karma(param, ne_init_k);
        end
    else
        for i_alpha = 1 : param.n_alpha
            alpha = param.Alpha(i_alpha);
            if alpha > 0.99 && alpha < 1
                ne_file_str = [ne_file_str, '_', num2str(alpha, '%.3f')];
            else
                ne_file_str = [ne_file_str, '_', num2str(alpha, '%.2f')];
            end
        end
        ne_file = [ne_file_str, '.mat'];
        
        load(ne_file, 'ne_param', 'ne_pi_down_u_k_up_b');
        K_ne{1} = ne_param.K;
        pi_ne{1} = ne_pi_down_u_k_up_b;
        pi_ne_pure{1} = get_pure_policy(pi_ne{1}, K_ne{1}, param);

        if param.karma_initialization == 2
            % Initialize karma as per stationary distribution predicted by NE
            % algorithm
            load(ne_file, 'ne_sigma_up_k');
            ne_init_k = get_init_k(ne_sigma_up_k, K_ne{1}, param);
        else
            ne_init_k = init_k;
        end
        k_ne{1} = allocate_karma(param, ne_init_k);
    end
end

% Social welfare karma policy distribution
if control.karma_sw_policy
    sw_file = 'karma_nash_equilibrium/results/sw_U_';
    for i_u = 1 : param.n_u
        sw_file = [sw_file, num2str(param.U(i_u)), '_'];
    end
    for i_mu = 1 : param.n_mu
    sw_file = [sw_file, 'phi', int2str(i_mu), '_'];
        for i_u = 1 : param.n_u
            for i_un = 1 : param.n_u
                sw_file = [sw_file, num2str(param.phi_down_mu_u_up_un(i_mu,i_u,i_un), '%.2f'), '_'];
            end
        end
    end
    sw_file = [sw_file, 'pay_', int2str(param.payment_rule),...
        '/k_bar_', num2str(param.k_bar, '%02d'), '.mat'];
    
    load(sw_file, 'ne_param', 'sw_pi_down_u_k_up_b');
    K_sw = ne_param.K;
    pi_sw = sw_pi_down_u_k_up_b;
    pi_sw_pure = get_pure_policy(pi_sw, K_sw, param);
    
    if param.karma_initialization == 2
        % Initialize karma as per stationary distribution predicted by SW
        % algorithm
        load(sw_file, 'sw_sigma_up_k');
        sw_init_k = get_init_k(sw_sigma_up_k, K_sw, param);
    else
        sw_init_k = init_k;
    end
    k_sw = allocate_karma(param, sw_init_k);
end

%% Number of times each agent was in an interaction
t_i = zeros(1, param.n_a);

%% Some memory allocations for the loop
this_u = zeros(1, 2);
this_i_u = zeros(1, 2);
this_k = zeros(1, 2);
this_b = zeros(1, 2);

%% Simulation run
% Convention:   win_id := id of agent that wins
%               lose_id := id of agent that loses
for t = 1 : param.T
    % Tell user where we are
    fprintf('Timestep: %d\n', t);

    % Sample agents i & j uniformly at random
    this_agents = datasample(agent_stream, agents, 2, 'Replace', false);

    % Increment number of interactions for picked agents
    this_t_i = t_i(this_agents) + 1;
    t_i(this_agents) = this_t_i;

    % Assert if an agent is picked too many times for memory allocated
    assert(max(this_t_i) < param.max_T_i,...
        'One agent was picked too many times. Increase maximum allowed number of interactions per agent.');

    % Urgency of sampled agents
    mu = agent_types(1,this_agents);
    this_i_u_last = i_u_last(this_agents);
    % See if we can sample urgency of both agents from same prob
    % distribution. This is faster than sampling per agent
    if mu(1) == mu(2) && (param.u_iid(mu(1)) || this_i_u_last(1) == this_i_u_last(2))
        this_i_u = datasample(u_stream, param.i_U, 2, 'Weights', squeeze(param.phi_down_mu_u_up_un(mu(1),this_i_u_last(1),:)));
    else
        for i_agent = 1 : 2
            this_i_u(i_agent) = datasample(u_stream, param.i_U, 1, 'Weights', squeeze(param.phi_down_mu_u_up_un(mu(i_agent),this_i_u_last(i_agent),:)));
        end
    end
    i_u_last(this_agents) = this_i_u;

    % Update history of urgency
    for i_agent = 1 : 2
        this_u(i_agent) = param.U(this_i_u(i_agent));
        u_hist(this_t_i(i_agent),this_agents(i_agent)) = this_u(i_agent);
    end


    %% RANDOM POLICY %%
    % Simply choose 'first' agent, since there is already randomization
    % in the agent order
    i_win = 1;

    % Update cost. Winning agent incurs 0 cost. Losing agent incurs
    % cost equal to their urgency
    for i_agent = 1 : 2
        if i_agent == i_win
            c_rand(this_t_i(i_agent),this_agents(i_agent)) = 0;
        else
            c_rand(this_t_i(i_agent),this_agents(i_agent)) = this_u(i_agent);
        end
    end

    %% CENTRALIZED POLICIES %%
    %% Centralized urgency policy
    % Agent with max urgency passes. Coin flip on tie
    % Find agent(s) with max urgency, which are candidates for passing
    [~, i_max_u] = multi_maxes(this_i_u);
    num_max_u = length(i_max_u);

    % Choose 'first' urgent agent (there is already randomization in
    % the agent order)
    i_win = i_max_u(1);

    % Update cost. Winning agent incurs 0 cost. Losing agent incurs
    % cost equal to their urgency
    for i_agent = 1 : 2
        if i_agent == i_win
            c_u(this_t_i(i_agent),this_agents(i_agent)) = 0;
        else
            c_u(this_t_i(i_agent),this_agents(i_agent)) = this_u(i_agent);
        end
    end

    %% Centralized accumulated cost policy
    % Agent with maximum accumulated cost (counting current urgency)
    % passes. Coin flip on tie
    a = accumulate_cost(c_a, this_agents, this_u, t_i, inf);

    [~, i_win] = max(a);

    % Update cost. Winning agent incurs 0 cost. Losing agent incurs
    % cost equal to their urgency
    for i_agent = 1 : 2
        if i_agent == i_win
            c_a(this_t_i(i_agent),this_agents(i_agent)) = 0;
        else
            c_a(this_t_i(i_agent),this_agents(i_agent)) = this_u(i_agent);
        end
    end

    %% Centralized urgency then accumulated cost policy
    % Agents with max urgency, which are candidates for passing, were
    % already found in first step of centralized urgency policy
    % If there are multiple agents with max urgency, pick one based on
    % accumulated cost like in centralized accumulated cost policy
    if num_max_u > 1
        a = accumulate_cost(c_u_a, this_agents, this_u, t_i, inf);
        [~, i_win] = max(a);
    else
        i_win = i_max_u;
    end

    % Update cost. Winning agent incurs 0 cost. Losing agent incurs
    % cost equal to their urgency
    for i_agent = 1 : 2
        if i_agent == i_win
            c_u_a(this_t_i(i_agent),this_agents(i_agent)) = 0;
        else
            c_u_a(this_t_i(i_agent),this_agents(i_agent)) = this_u(i_agent);
        end
    end

    %% Fairness horizon policies
    if control.fairness_horizon_policies
        for i_fair_hor = 1 : param.n_fairness_horizon
            %% Centralized accumulated cost with fairness horizon
            % Agent with maximum accumulated cost up to fairness horizon
            % (counting current urgency) passes
            % Coin flip on tie
            a = accumulate_cost(c_fair_hor_a{i_fair_hor}, this_agents, this_u, t_i, param.fairness_horizon(i_fair_hor));
            [~, i_win] = max(a);

            % Update cost. Winning agent incurs 0 cost. Losing agent
            % incurs cost equal to their urgency
            for i_agent = 1 : 2
                if i_agent == i_win
                    c_fair_hor_a{i_fair_hor}(this_t_i(i_agent),this_agents(i_agent)) = 0;
                else
                    c_fair_hor_a{i_fair_hor}(this_t_i(i_agent),this_agents(i_agent)) = this_u(i_agent);
                end
            end

            %% Centralized urgency then accumulated cost with fairness horizon
            % Agents with max urgency, which are candidates for passing,
            % were already found in first step of centralized urgency policy
            % If there are multiple agents with max urgency, pick one based
            % on accumulated cost up to fairness horizon
            if num_max_u > 1
                a = accumulate_cost(c_fair_hor_u_a{i_fair_hor}, this_agents, this_u, t_i, param.fairness_horizon(i_fair_hor));
                [~, i_win] = max(a);
            else
                i_win = i_max_u;
            end

            % Update cost. Winning agent incurs 0 cost. Losing agent
            % incurs cost equal to their urgency
            for i_agent = 1 : 2
                if i_agent == i_win
                    c_fair_hor_u_a{i_fair_hor}(this_t_i(i_agent),this_agents(i_agent)) = 0;
                else
                    c_fair_hor_u_a{i_fair_hor}(this_t_i(i_agent),this_agents(i_agent)) = this_u(i_agent);
                end
            end
        end
    end

    %% KARMA POLICIES %%
    % Save state of random stream to revert to before randomization of
    % the different policies. This ensures 'ties' are handled the same
    karma_stream_state = karma_stream.State;

    %% Nash equilibrium karma policies
    if control.karma_ne_policies
        for i_alpha_comp = 1 : param.n_alpha_comp
            % Get agents' bids from their policies
            for i_agent = 1 : 2
                this_k(i_agent) = k_ne{i_alpha_comp}(this_t_i(i_agent),this_agents(i_agent));
                i_k = min([this_k(i_agent) + 1, length(K_ne{i_alpha_comp})]); % Uses policy of k_max if it is exceeded
                this_b(i_agent) = pi_ne_pure{i_alpha_comp}(this_i_u(i_agent),i_k);
                if isnan(this_b(i_agent))
                    karma_stream.State = karma_stream_state; % Reset karma stream to current state for consitency across policies
                    this_b(i_agent) = datasample(karma_stream, K_ne{i_alpha_comp}, 1, 'Weights', squeeze(pi_ne{i_alpha_comp}(this_i_u(i_agent),i_k,:)));
                end
            end

            % Agent bidding max karma passes and pays karma bidded
            [b_win, i_win] = max(this_b);
            
            if param.payment_rule
                % Pay difference
                b_lose = min(this_b);
                p = b_win - b_lose;
            else
                % Pay as bid
                p = b_win;
            end
            
            % Update cost and karma. Winning agent incurs 0 cost and pays
            % karma. Losing agent incurs cost equal to their urgency and
            % gets karma
            for i_agent = 1 : 2
                if i_agent == i_win
                    c_ne{i_alpha_comp}(this_t_i(i_agent),this_agents(i_agent)) = 0;
                    k_ne{i_alpha_comp}(this_t_i(i_agent)+1,this_agents(i_agent)) = this_k(i_agent) - p;
                else
                    c_ne{i_alpha_comp}(this_t_i(i_agent),this_agents(i_agent)) = this_u(i_agent);
                    k_ne{i_alpha_comp}(this_t_i(i_agent)+1,this_agents(i_agent)) = this_k(i_agent) + p;
                end
            end
        end
    end

    %% Social welfare karma policy
    if control.karma_sw_policy
        % Get agents' bids from their policies
        for i_agent = 1 : 2
            this_k(i_agent) = k_sw(this_t_i(i_agent),this_agents(i_agent));
            i_k = min([this_k(i_agent) + 1, length(K_sw)]); % Uses policy of k_max if it is exceeded
            this_b(i_agent) = pi_sw_pure(this_i_u(i_agent),i_k);
            if isnan(this_b(i_agent))
                karma_stream.State = karma_stream_state; % Reset karma stream to current state for consitency across policies
                this_b(i_agent) = datasample(karma_stream, K_sw, 1, 'Weights', squeeze(pi_sw(this_i_u(i_agent),i_k,:)));
            end
        end

        % Agent bidding max karma passes and pays karma bidded
        [b_win, i_win] = max(this_b);
        
        if param.payment_rule
            % Pay difference
            b_lose = min(this_b);
            p = b_win - b_lose;
        else
            % Pay as bid
            p = b_win;
        end

        % Update cost and karma. Winning agent incurs 0 cost and pays
        % karma. Losing agent incurs cost equal to their urgency and gets
        % karma
        for i_agent = 1 : 2
            if i_agent == i_win
                c_sw(this_t_i(i_agent),this_agents(i_agent)) = 0;
                k_sw(this_t_i(i_agent)+1,this_agents(i_agent)) = this_k(i_agent) - p;
            else
                c_sw(this_t_i(i_agent),this_agents(i_agent)) = this_u(i_agent);
                k_sw(this_t_i(i_agent)+1,this_agents(i_agent)) = this_k(i_agent) + p;
            end
        end
    end
end

%% Perfromance measures
fprintf('Computing performance measures\n');

%% Accumulated costs per agent at each time step
fprintf('Computing accumulated costs\n');
a_rand = get_accumulated_cost(c_rand, t_i, param);
a_u = get_accumulated_cost(c_u, t_i, param);
a_a = get_accumulated_cost(c_a, t_i, param);
a_u_a = get_accumulated_cost(c_u_a, t_i, param);
if control.fairness_horizon_policies
    a_fair_hor_a = cell(param.n_fairness_horizon, 1);
    a_fair_hor_u_a = cell(param.n_fairness_horizon, 1);
    for i_fair_hor = 1 : param.n_fairness_horizon
        a_fair_hor_a{i_fair_hor} = get_accumulated_cost(c_fair_hor_a{i_fair_hor}, t_i, param);
        a_fair_hor_u_a{i_fair_hor} = get_accumulated_cost(c_fair_hor_u_a{i_fair_hor}, t_i, param);
    end
end
if control.karma_ne_policies
    a_ne = cell(param.n_alpha_comp, 1);
    for i_alpha_comp = 1 : param.n_alpha_comp
        a_ne{i_alpha_comp} = get_accumulated_cost(c_ne{i_alpha_comp}, t_i, param);
    end
end
if control.karma_sw_policy
    a_sw = get_accumulated_cost(c_sw, t_i, param);
end

%% Inefficiency vs. time
fprintf('Computing efficiencies\n');
IE_rand = mean(a_rand, 2);
IE_u = mean(a_u, 2);
IE_a = mean(a_a, 2);
IE_u_a = mean(a_u_a, 2);
if control.fairness_horizon_policies
    IE_fair_hor_a = cell(param.n_fairness_horizon, 1);
    IE_fair_hor_u_a = cell(param.n_fairness_horizon, 1);
    for i_fair_hor = 1 : param.n_fairness_horizon
        IE_fair_hor_a{i_fair_hor} = mean(a_fair_hor_a{i_fair_hor}, 2);
        IE_fair_hor_u_a{i_fair_hor} = mean(a_fair_hor_u_a{i_fair_hor}, 2);
    end
end
if control.karma_ne_policies
    IE_ne = cell(param.n_alpha_comp, 1);
    for i_alpha_comp = 1 : param.n_alpha_comp
        IE_ne{i_alpha_comp} = mean(a_ne{i_alpha_comp}, 2);
    end
end
if control.karma_sw_policy
    IE_sw = mean(a_sw, 2);
end

%% Unfairness vs. time
fprintf('Computing fairness\n');
UF_rand = var(a_rand, [], 2);
UF_u = var(a_u, [], 2);
UF_a = var(a_a, [], 2);
UF_u_a = var(a_u_a, [], 2);
if control.fairness_horizon_policies
    UF_fair_hor_a = cell(param.n_fairness_horizon, 1);
    UF_fair_hor_u_a = cell(param.n_fairness_horizon, 1);
    for i_fair_hor = 1 : param.n_fairness_horizon
        UF_fair_hor_a{i_fair_hor} = var(a_fair_hor_a{i_fair_hor}, [], 2);
        UF_fair_hor_u_a{i_fair_hor} = var(a_fair_hor_u_a{i_fair_hor}, [], 2);
    end
end
if control.karma_ne_policies
    UF_ne = cell(param.n_alpha_comp, 1);
    for i_alpha_comp = 1 : param.n_alpha_comp
        UF_ne{i_alpha_comp} = var(a_ne{i_alpha_comp}, [], 2);
    end
end
if control.karma_sw_policy
    UF_sw = var(a_sw, [], 2);
end

%% Karma distributions
if control.karma_ne_policies
    fprintf('Computing karma distributions\n');
    k_ne_dist = cell(param.n_alpha_comp, 1);
    k_ne_dist_agents = cell(param.n_alpha_comp, 1);
    for i_alpha_comp = 1 : param.n_alpha_comp
        [k_ne_dist{i_alpha_comp}, k_ne_dist_agents{i_alpha_comp}] = get_karma_dist(k_ne{i_alpha_comp}, param);
    end
end
if control.karma_sw_policy
    [k_sw_dist, k_sw_dist_agents] = get_karma_dist(k_sw, param);
end

%% Entropy
if control.compute_entropy
    fprintf('Computing entropy\n');
    % Entropy of fairness horizon accumulated costs
    if control.fairness_horizon_policies
        ent_fair_hor_rand = zeros(param.n_fairness_horizon, 1);
        ent_fair_hor_u = zeros(param.n_fairness_horizon, 1);
        ent_fair_hor_a = zeros(param.n_fairness_horizon, 1);
        ent_fair_hor_u_a = zeros(param.n_fairness_horizon, 1);
        for i_fair_hor = 1 : param.n_fairness_horizon
            ent_fair_hor_rand(i_fair_hor) = get_entropy_fairness_horizon(c_rand, t_i, param.fairness_horizon(i_fair_hor), param);
            ent_fair_hor_u(i_fair_hor) = get_entropy_fairness_horizon(c_u, t_i, param.fairness_horizon(i_fair_hor), param);
            ent_fair_hor_a(i_fair_hor) = get_entropy_fairness_horizon(c_fair_hor_a{i_fair_hor}, t_i, param.fairness_horizon(i_fair_hor), param);
            ent_fair_hor_u_a(i_fair_hor) = get_entropy_fairness_horizon(c_fair_hor_u_a{i_fair_hor}, t_i, param.fairness_horizon(i_fair_hor), param);
        end
    end
    
    % Entropy of karma
    if control.karma_ne_policies
        ent_ne = zeros(param.n_alpha_comp, 1);
        for i_alpha_comp = 1 : param.n_alpha_comp
            ent_ne(i_alpha_comp) = get_entropy(k_ne_dist{i_alpha_comp});
        end
    end
    if control.karma_sw_policy
        ent_sw = get_entropy(k_sw_dist);
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
fprintf('Plotting\n');
do_plots;

%% Inform user when done and sound
fprintf('DONE\n\n');
load mtlb.mat;
sound(mtlb);