function param = load_parameters()
% Population size
param.N = 200;

% Vector of all urgency values. Make sure it's sorted
param.U = [1; 10];
param.U = sort(param.U);

% 'Low' urgency
param.u_min = param.U(1);

% Number of urgency values
param.num_U = length(param.U);

% Transition matrix for urgency Markov chain
param.mu_down_u_up_un = zeros(param.num_U, param.num_U);
% Next urgency is skewed towards current urgency
param.mu_bias = 0.5;
for i_u = 1 : param.num_U
    for i_un = 1 : param.num_U
        if i_un == i_u
            param.mu_down_u_up_un(i_u,i_un) = param.mu_bias;
        else
            param.mu_down_u_up_un(i_u,i_un) = (1 - param.mu_bias) / (param.num_U - 1);
        end
    end
end
% param.mu_bias = nan;     % Indicates general Markov chain
% param.mu_down_u_up_un = [0.8, 0.2; 0.4, 0.6];

% Stationary distribution of urgency Markov chain
param.p_up_u = func.stat_dist(param.mu_down_u_up_un);

% Number of agents in one interaction
param.I_size = 2;

% Number of agents that win
param.num_win = 1;

% Number of agents that lose
param.num_lose = param.I_size - param.num_win;

% Number of days. Urgency of agents stays contant during day
param.num_days = 100000;

% Number of interactions per day
param.num_inter_per_day = 1;

% Total number of interactions in simulation
param.tot_num_inter = param.num_days * param.num_inter_per_day;

% This flag ensures number of interactions is the same at the end of the
% simulation; i.e. cycles through agents when picking interaction pairs
param.same_num_inter = false;

% If number of interactions is to be the same, make sure there is a whole
% number of interaction pairs in N, as well as a whole number of
% interaction sets in the number of iterations. This is not required if one
% is to make sure everything divides out evenly
if param.same_num_inter
    param.num_inter_in_N = round(param.N / param.I_size);
    param.N = param.num_inter_in_N * param.I_size;
    param.tot_num_inter = round(param.tot_num_inter / param.num_inter_in_N) * param.num_inter_in_N;
    param.num_inter_per_day = round(param.tot_num_inter / param.num_days);
    param.tot_num_inter = param.num_days * param.num_inter_per_day;
end

% Warm up days. Accumulated cost resets after warm-up
param.warm_up_days = 0;

% Timestep at end of warm up days
param.t_warm_up = param.warm_up_days * param.num_inter_per_day;

% Expected total number of interactions per agent
param.num_inter_per_agent = round(param.tot_num_inter / (param.N / param.I_size));

% Maximum allowed number of interactions per agent. Used in memory
% allocations
param.max_num_inter_per_agent = round(1.5 * param.num_inter_per_agent);

% Flag to normalize all costs based on how many times agents have been
% picked to interact
param.normalize_cost = true;

% Standardization method for accumulated cost
% 0 => 0-mean 1-variance standardization
% 1 => order ranking standardization
% 2 => normalized order ranking standardization, i.e. order ranking scaled
% between 0-1
param.standardization_method = 2;

% Limited memory policies steps
% param.lim_mem_steps = 2.^(0 : ceil(log2(param.num_inter_per_agent)));
param.lim_mem_steps = [1 : 10, 2.^(4 : ceil(log2(param.num_inter_per_agent)))];

% Limited memory number of steps
param.num_lim_mem_steps = length(param.lim_mem_steps);

% Average karma level(s)
% param.k_ave = 11 : 12;
param.k_ave = 10;

% Total karma in the system
param.k_tot = param.k_ave * param.N;

% Message exchange method
% 0 => Pay as bid
% 1 => Pay difference
% 2 => Pay difference and pay 1 on tie
param.m_exchange = 0;

% Future discount factor(s)
% param.alpha = [1.00 : -0.001 : 0.991, 0.99 : -0.01 : 0.96, 0.95 : -0.05 : 0.10];
param.alpha = 0.97;

% Number of future discount factor(s)
param.num_alpha = length(param.alpha);

% Karma initialization method
% 0 => Initialize all policies with all agents having k_ave
% 1 => Initialize all policies with the same initial karma, as per uniform
% distribution (modified to have average karma k_ave)
% 2 => Initialize policies with their respective predicted stationary
% distribution
param.karma_initialization = 2;

% Tolerance to consider policy 'pure'
% If ration between 2nd highest and highest probability is lower,
% policy is considered pure in highest probability message
param.pure_policy_tol = 5e-2;

% Save results
param.save = true;

% Plot flags
% Global plot flag
param.plot = true;
% Flag to plot accumulated costs
param.plot_a = false;
% Flag to plot fairness vs. time
param.plot_F = false;
% Flag to plot standardized accumulated costs
param.plot_a_std = false;
% Flag to plot accumulated costs autocorrelation
param.plot_a_acorr = false;

end