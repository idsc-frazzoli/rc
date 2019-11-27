function ne_param = load_ne_parameters()
% Alpha
ne_param.alpha = 0.8;
% Tolerance for convergence of (D,T) pair
ne_param.D_T_tol = 1e-4;

% Maximum number of iterations for convergence of (D,T) pair
ne_param.D_T_max_iter = 100;

% Momentum
ne_param.tau = 1.0;

% Tolerance for convergence of policy
ne_param.policy_tol = 1e-3;

% Maximum number of policy iterations
ne_param.policy_max_iter = 10;

% Maximum number of Nash Equilibrium policy iterations
ne_param.ne_policy_max_iter = 1000;
end