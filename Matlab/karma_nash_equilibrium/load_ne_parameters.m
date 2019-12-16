function ne_param = load_ne_parameters()
% Load main parameters
param = load_parameters();

% Vector of all urgency values
ne_param.U = param.U;

% Vector of probabilities of urgency values. This must sum to 1
ne_param.p_U = param.p_U;

% Number of urgency values
ne_param.num_U = param.num_U;

% Low urgency
ne_param.u_low = param.u_low;

% High urgency
ne_param.u_high = param.u_high;

% Vector of all karma values
ne_param.K = (param.k_min : 1 : param.k_max).';

% Number of karma values
ne_param.num_K = length(ne_param.K);

% Maximum karma
ne_param.k_max = param.k_max;

% Average karma
ne_param.k_ave = param.k_ave;

% Number of states, which is number of urgency * number of karma values
ne_param.num_X = ne_param.num_U * ne_param.num_K;

% Alpha
ne_param.alpha = 0 : 0.05 : 0.95;
% ne_param.alpha = 0.75;

% Tolerance for convergence of (D,T) pair
ne_param.D_T_tol = 1e-3;

% Maximum number of iterations for convergence of (D,T) pair
ne_param.D_T_max_iter = 100;

% Tolerance for convergence of D
ne_param.D_tol = 1e-3;

% Maximum number of iterations for convergence of D
ne_param.D_max_iter = 1000;

% Tolerance for convergence of V
ne_param.V_tol = 1e-3;

% Maximum number of iterations for convergence of V
ne_param.V_max_iter = 1000;

% Momentum on policy
ne_param.policy_tau = 0.05;

% Momentum on stationary distribution
ne_param.D_tau = 1.0;

% Tolerance for convergence of policy
ne_param.policy_tol = 1e-4;

% Maximum number of policy iterations
ne_param.policy_max_iter = 100;

% Maximum number of Nash Equilibrium policy iterations
ne_param.ne_policy_max_iter = 1000;

% Do plots
ne_param.plot = false;

% Save results
ne_param.save = false;

end