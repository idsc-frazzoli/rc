function param = load_parameters()
% Population size
param.N = 200;

% Vector of all urgency values
param.U = [0; 3];

% Vector of probabilities of urgency values. This must sum to 1
param.p_U = [0.5; 0.5];

% Number of urgency values
param.num_U = length(param.U);

% Transition matrix for urgency markov process
param.mu_down_u_up_un = zeros(param.num_U, param.num_U);
for i_u = 1 : param.num_U
    param.mu_down_u_up_un(i_u,:) = param.p_U.';
end

% Low urgency
param.u_low = min(param.U);

% High urgency
param.u_high = max(param.U);

% Number of agents in one intersection
param.I_size = 2;

% Number of agents that win
param.num_win = 1;

% Number of agents that lose
param.num_lose = param.I_size - param.num_win;

% Number of days. Urgency of agents stays contant during day
param.num_days = 10000;

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

% Total expected number of interactions per agent
param.num_inter_per_agent = param.tot_num_inter / (param.N / param.I_size);

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
param.lim_mem_steps = 2.^(0 : ceil(log2(param.num_inter_per_agent)));

% Limited memory number of steps
param.num_lim_mem_steps = length(param.lim_mem_steps);

% Minimum karma level
param.k_min = 0;

% Maximum karma level
param.k_max = 12;

% Vector of all karma values
param.K = (param.k_min : param.k_max).';

% Average karma level(s)
% param.k_ave = 0 : 12;
param.k_ave = 6;

% Message discretization interval
param.m_interval = 1.0;

% Vector of all message values
param.M = param.k_min : param.m_interval : param.k_max;

% Future discount factor(s)
param.alpha = [0 : 0.05 : 0.95, 1 - eps];
% param.alpha = 1 - eps;

% Number of future discount factor(s)
param.num_alpha = length(param.alpha);

end