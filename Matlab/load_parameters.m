function param = load_parameters()

%% Setting parameters
% Average amount of karma
param.k_bar = 10;
% Set to row vector to perform multiple SW computations in one shot
% param.k_bar = 11 : 12;

% Number of urgency types
param.n_mu = 1;

% Future awareness types
param.Alpha = 0.97;
% Set to row vector to simulate multiple alpha values or perform multiple
% NE computations in one shot
% param.Alpha = [1.00 : -0.001 : 0.991, 0.99 : -0.01 : 0.96, 0.95 : -0.05 : 0.10];
% Set to column vector for multiple future awareness types
% param.Alpha = [0.30; 0.97];
valid_Alpha = (size(param.Alpha, 1) == 1 || size(param.Alpha, 2) == 1) ...
    && min(param.Alpha) >= 0 && max(param.Alpha) <= 1;
assert(valid_Alpha, 'Invalid future awareness types');

% Number of future awareness types
param.n_alpha = size(param.Alpha, 1);

% Number of future awareness values for simulation/computation
param.n_alpha_comp = size(param.Alpha, 2);

% Type distribution
param.g = zeros(param.n_mu, param.n_alpha);
param.g(1,1) = 1.0;
valid_g = min(param.g(:)) >= 0 && sum(param.g(:)) == 1;
assert(valid_g, 'Invalid type distribution');

% Sorted set of urgency values
param.U = [1; 10];
param.U = sort(param.U);

% Number of urgency values
param.n_u = length(param.U);

% Urgency Markov chain
param.phi_down_mu_u_up_un = zeros(param.n_mu, param.n_u, param.n_u);
param.phi_down_mu_u_up_un(1,:,:) = [0.50, 0.50;
                                    0.50, 0.50];
valid_phi = true;
for i_mu = 1 : param.n_mu
    for i_u = 1 : param.n_u
        valid_phi = valid_phi ...
            && min(param.phi_down_mu_u_up_un(i_mu,i_u,:)) >= 0 ...
            && sum(param.phi_down_mu_u_up_un(i_mu,i_u,:)) == 1;
        if ~valid_phi
            break;
        end
    end
    if ~valid_phi
        break;
    end
end
assert(valid_phi, 'Invalid urgency Markov chain');

% Stationary distribution of urgency Markov chain
param.prob_down_mu_up_u = zeros(param.n_mu, param.n_u);
for i_mu = 1 : param.n_mu
    param.prob_down_mu_up_u(i_mu,:) = func.stat_dist(squeeze(param.phi_down_mu_u_up_un(i_mu,:,:)));
end

% Payment rule
% 0 => Pay as bid
% 1 => Pay difference
param.payment_rule = 0;

%% Simulation parameters
% Number of agents
param.n_a = 200;

% Total amount of karma
param.k_tot = param.k_bar * param.n_a;

% Number of timesteps
param.T = 100000;

% Expected number of interactions per agent
% 2 agents engage in an interacion, which doubles their chances
param.exp_T_i = round(param.T / (param.n_a / 2));

% Maximum allowed number of interactions per agent. Used in memory
% allocations
param.max_T_i = round(1.5 * param.exp_T_i);

% Fairness horizons
param.fairness_horizon = [1 : 10, 2.^(4 : ceil(log2(param.exp_T_i)))];

% Number of fairness horizons
param.n_fairness_horizon = length(param.fairness_horizon);

% Karma initial distribution in simulation
% 0 => All agents have average karma k_bar
% 1 => Uniform distribution over [0 : 2 * k_bar]
% 2 => Stationary distribution predicted by NE/SW computation
param.karma_initialization = 1;

% Tolerance to consider policy 'pure'
% If ratio between 2nd highest and highest probability is lower,
% policy is considered pure in highest probability message
param.pure_policy_tol = 5e-2;

% Save results
param.save = true;

end