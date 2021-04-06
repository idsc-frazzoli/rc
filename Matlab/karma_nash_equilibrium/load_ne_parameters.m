function ne_param = load_ne_parameters(param)

% Maximum karma level. Used as a starting point
ne_param.k_max = 50;

% Set of all karma values
ne_param.K = (0 : ne_param.k_max).';

% Number of karma values
ne_param.n_k = length(ne_param.K);

% Where is k_bar
ne_param.i_k_bar = find(ne_param.K == param.k_bar);

% Number of states
ne_param.n_x = param.n_u * ne_param.n_k;

% Set of output values
ne_param.O = [0; 1];

% Number of output values
ne_param.n_o = length(ne_param.O);

% Policy initialization
% 0 => Bid urgency
% 1 => Bid 0.5 * u / u_max * k (~ 'bid half if urgent')
% 2 => Bid 1 * u / u_max * k (~ 'bid all if urgent')
% 3 => Bid random
ne_param.policy_initialization = 0;

% Karma distribution initialization
% 0 => All agents have average karma k_bar
% 1 => Uniform distribution over [0 : 2 * k_bar]
ne_param.karma_initialization = 0;

% Tolerance for convergence of population distribution
ne_param.ne_d_tol = 1e-6;

% Maximum number of iterations for convergence of population distribution
ne_param.ne_d_max_iter = 100000;

% Momentum on population distribution
ne_param.ne_d_mom = 0.05;      % Default value
% ne_param.ne_d_mom = 1.00;    % Use for alpha = 1

% Tolerance for maximum probability of k_max
ne_param.max_sigma_k_max = 1e-4;  % Use for NE computation
% ne_param.max_sigma_k_max = 1e-3;    % Use for SW computation

% Step size for increasing k_max on saturation detection
ne_param.k_max_step = 5;

% Maximum k_max allowable. Code asserts if this needs to be exceeded
ne_param.max_k_max = 120;

% Tolerance for convergence of value function
ne_param.J_tol = 1e-10;

% Maximum number of iterations for convergence of value function
ne_param.J_max_iter = 10000;

% Tolerance for best response deviation on value function
ne_param.br_J_tol = 1e-4;

% Tolerance for convergence of Nash equilibrium policy
ne_param.ne_pi_tol = 1e-6;

% Maximum number of Nash equilibrium policy iterations
ne_param.ne_pi_max_iter = 1000;

% Momentum on Nash equilibrium policy
ne_param.ne_pi_mom = 0.05;

% Keep history of policy evolution
ne_param.store_hist = false;

% Limit cycle check horizon
ne_param.limit_cycle_horizon = 0;

% Do plots
ne_param.plot = false;

% Save results
ne_param.save = true;

end