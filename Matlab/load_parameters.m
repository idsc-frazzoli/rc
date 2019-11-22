function param = load_parameters()
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

% Limited memory policies steps
param.lim_mem_steps = 2.^(0 : ceil(log2(param.num_inter_per_agent)));

% Limited memory number of steps
param.lim_mem_num_steps = length(param.lim_mem_steps);

% Minimum karma level
param.k_min = 0;

% Maximum karma level
param.k_max = 12;
end

