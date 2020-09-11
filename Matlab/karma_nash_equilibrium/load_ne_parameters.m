function ne_param = load_ne_parameters()
% Load main parameters
param = load_parameters();

% Vector of all urgency values
ne_param.U = param.U;

% Number of urgency values
ne_param.num_U = param.num_U;

% Transition matrix for urgency markov chain
ne_param.mu_down_u_up_un = param.mu_down_u_up_un;

% Stationary distribution of urgency Markov chain
ne_param.p_up_u = param.p_up_u;

% Average karma
ne_param.k_ave = param.k_ave;

% Maximum karma level. Used as a starting point
ne_param.k_max = 50;

% Vector of all karma values
ne_param.K = (0 : ne_param.k_max).';

% Number of karma values
ne_param.num_K = length(ne_param.K);

% Where is k_ave
ne_param.i_k_ave = find(ne_param.K == ne_param.k_ave);

% Number of states
ne_param.num_X = ne_param.num_U * ne_param.num_K;

% Message exchange method
% 0 => Pay as bid
% 1 => Pay difference
% 2 => Pay difference and pay 1 on tie
ne_param.m_exchange = param.m_exchange;

% Vector of output values
ne_param.O = [0; 1];

% Number of output values
ne_param.num_O = length(ne_param.O);

% Future discount factor(s)
ne_param.alpha = param.alpha;

% Number of future discount factor(s)
ne_param.num_alpha = param.num_alpha;

% Initial policy for algorithms
% 0 => Bid urgency
% 1 => Bid 0.5 * u / u_max * k (~ 'bid half if urgent')
% 2 => Bid 1 * u / u_max * k (~ 'bid all if urgent')
% 3 => Bid random
ne_param.pi_init_method = 0;

% Tolerance for convergence of stationary distribution
ne_param.d_tol = 1e-7;

% Maximum number of iterations for convergence of stationary distribution
ne_param.d_max_iter = 100000;

% Tolerance for maximum probability of k_max
ne_param.max_s_k_max = 1e-4;  % Use for NE computation
% ne_param.max_s_k_max = 1e-3;    % Use for SW computation

% Step size for increasing k_max on saturation detection
ne_param.k_max_step = 5;

% Maximum k_max allowable. Code asserts if this needs to be exceeded
ne_param.max_k_max = 120;

% Momentum on stationary distribution
ne_param.d_mom = 0.05;      % Default value
% ne_param.d_mom = 1.00;    % Use for alpha = 1

% Tolerance for convergence of v
ne_param.v_tol = 1e-10;

% Maximum number of iterations for convergence of v
ne_param.v_max_iter = 10000;

% Tolerance for best response deviation on v
% ne_param.br_v_tol = 1e-4;
ne_param.br_v_tol = realmin;

% Tolerance for convergence of best response policy
ne_param.br_pi_tol = 1e-4;

% Maximum number of best response policy iterations
ne_param.br_pi_max_iter = 1;

% Tolerance for convergence of Nash equilibrium policy
ne_param.ne_pi_tol = 1e-7;

% Maximum number of Nash equilibrium policy iterations
ne_param.ne_pi_max_iter = 10000;

% Logit constant for perturbed best response (logit dynamics)
ne_param.logit_const = 0.025;

% Momentum on Nash equilibrium policy
ne_param.ne_pi_mom = 0.05 * ones(1, ne_param.ne_pi_max_iter);

% Keep history of policy evolution
ne_param.store_hist = false;

% Limit cycle check horizon
ne_param.limit_cycle_horizon = 0;

% Controls smoothing of winning and karma exchange probabilities
% 0 => No smoothing
% 1 => Smoothing using normal distributions
% 2 => Smoothing using logistic distributions
ne_param.smoothing = 0;

% Smoothing factor for winning probability
ne_param.gamma_s = 0.75;

% Smoothing factor for karma exchange probability
ne_param.beta_s = 0.5;

% Do plots
ne_param.plot = false;

% Save results
ne_param.save = true;

end