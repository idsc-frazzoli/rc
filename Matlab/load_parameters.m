function param = load_parameters()

%% Setting parameters
% Average amount of karma
param.k_bar = 10;
% Set to row vector to perform multiple SW computations in one shot
% param.k_bar = 11 : 12;

% Number of urgency types
param.n_mu = 1;

% Future awareness types
% param.Alpha = 0.97;
% Set to row vector to simulate multiple alpha values or perform multiple
% NE computations in one shot
param.Alpha = [1.00 : -0.01 : 0.96, 0.95 : -0.05 : 0.10, 0.00];
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
% We only consider independant urgency and future awareness type
% distributions for now
% Urgency type distribution
param.w_up_mu = 1.0;
valid_w = length(param.w_up_mu) == param.n_mu ...
    && min(param.w_up_mu) >= 0 && sum(param.w_up_mu) == 1;
assert(valid_w, 'Invalid urgency type distribution');
% Future awareness type distribution
param.z_up_alpha = 1;
% param.z_up_alpha = [0.1; 0.9];
valid_z = length(param.z_up_alpha) == param.n_alpha ...
    && min(param.z_up_alpha) >= 0 && sum(param.z_up_alpha) == 1;
assert(valid_z, 'Invalid future awareness type distribution');
% Joint type distribution
param.g_up_mu_alpha = reshape(outer(param.w_up_mu, param.z_up_alpha), param.n_mu, []);

% Sorted set of urgency values
param.U = [1; 10];
param.U = sort(param.U);

% Number of urgency values
param.n_u = length(param.U);

% Index set of urgency values
param.i_U = 1 : param.n_u;

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
% Also detect if urgency is i.i.d, in which case we can simply sample from
% stationary distribution. This is the case when all rows of the transition
% matrix are equal to the stationary distribution
param.prob_down_mu_up_u = zeros(param.n_mu, param.n_u);
param.u_iid = true(param.n_mu, 1);
for i_mu = 1 : param.n_mu
    param.prob_down_mu_up_u(i_mu,:) = stat_dist(squeeze(param.phi_down_mu_u_up_un(i_mu,:,:)));
    
    for i_u = 1 : param.n_u
        if norm(param.prob_down_mu_up_u(i_mu,:).' - squeeze(param.phi_down_mu_u_up_un(i_mu,i_u,:)), inf) >= eps
            param.u_iid(i_mu) = false;
            break;
        end
    end
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
param.karma_initialization = 2;

% Tolerance to consider policy 'pure'
% If ratio between 2nd highest and highest probability is lower,
% policy is considered pure in highest probability message
param.pure_policy_tol = 5e-2;

% Save results
param.save = false;

end