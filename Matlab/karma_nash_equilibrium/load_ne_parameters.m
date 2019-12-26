function ne_param = load_ne_parameters()
% Load main parameters
param = load_parameters();

% Vector of all urgency values
ne_param.U = param.U;

% Vector of probabilities of urgency values. This must sum to 1
ne_param.p_U = param.p_U;

% Number of urgency values
ne_param.num_U = param.num_U;

% Transition matrix for urgency markov process
ne_param.mu_down_u_up_un = param.mu_down_u_up_un;

% Low urgency
ne_param.u_low = param.u_low;

% High urgency
ne_param.u_high = param.u_high;

% Vector of all karma values
ne_param.K = (param.k_min : param.k_max).';

% Number of karma values
ne_param.num_K = length(ne_param.K);

% Minimum karma
ne_param.k_min = param.k_min;

% Maximum karma
ne_param.k_max = param.k_max;

% Average karma
ne_param.k_ave = param.k_ave;

% Partial messages
r_interval = 0.25;
ne_param.R = 0 : r_interval : 1 - r_interval;

% Number of partial messages
ne_param.num_R = length(ne_param.R);

% Number of states, which is number of urgency * number of karma values
ne_param.num_X = ne_param.num_U * ne_param.num_K;

% Future discount factor(s)
%ne_param.alpha = 0 : 0.05 : 0.95;
ne_param.alpha = 0.75;

% Tolerance for convergence of stationary distribution
ne_param.d_tol = 1e-3;

% Maximum number of iterations for convergence of stationary distribution
ne_param.d_max_iter = 1000;

% Momentum on stationary distribution
ne_param.d_mom = 1.0;

% Tolerance for convergence of v
ne_param.v_tol = 1e-3;

% Maximum number of iterations for convergence of v
ne_param.v_max_iter = 1000;

% Tolerance for best response deviation on v
ne_param.br_v_tol = 1e-8;
% ne_param.br_v_tol = 1e-2;

% Tolerance for convergence of best response policy
ne_param.br_pi_tol = 1e-4;

% Maximum number of best response policy iterations
ne_param.br_pi_max_iter = 100;

% Tolerance for convergence of Nash equilibrium policy
ne_param.ne_pi_tol = 1e-5;

% Maximum number of Nash equilibrium policy iterations
ne_param.ne_pi_max_iter = 1000;

% Momentum on Nash equilibrium policy
ne_param.ne_pi_mom = 0.05;

% Do plots
ne_param.plot = false;

% Save results
ne_param.save = false;

end