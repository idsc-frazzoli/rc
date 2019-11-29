clear;
close all;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
default_width = screenwidth / 2;
default_height = screenheight / 3;
load('karma_nash_equilibrium/RedColormap.mat');

%% Parameters
% NE computation parameters
ne_param = load_ne_parameters();

%% Iterative algorithm to find Karma Nash equilibrium %%

%% Step 0: Initialization
% Step 0.1: NE policy matrix for non-zero urgency guess
% Parametrized as a cell matrix. Row i in cell corresponds to policy for
% karma level k(i) and has only i columns since agents cannot bid more than
% their karma. Column j in row i denotes probability of transmitting
% message k(j) when karma level is k(i)
% Note 1: All entries are non-negative and rows sum to 1
% Note 2: NE policy for urgency 0 is to bid 0 karma always. This is
% hardcoded in algorithm and not parametrized
% Initialize to uniform distribution
policy = cell(ne_param.num_k, 1);
% for i_k_i = 1 : ne_param.num_k
%     policy{i_k_i} = 1 / i_k_i * ones(1, i_k_i);
% end
% for i_k_i = 1 : ne_param.num_k
%     policy{i_k_i} = rand(1, i_k_i);
%     policy{i_k_i} = policy{i_k_i} / sum(policy{i_k_i});
% end
% for i_k_i = 1 : ne_param.num_k
%     policy{i_k_i} = zeros(1, i_k_i);
%     policy{i_k_i}(i_k_i) = 1;
% end
% for i_k_i = 1 : ne_param.num_k
%     policy{i_k_i} = zeros(1, i_k_i);
%     policy{i_k_i}(1) = 1;
% end
for i_k_i = 1 : ne_param.num_k
    policy{i_k_i} = zeros(1, i_k_i);
end
policy{1}(1) = 1;
policy{2}(2) = 1;
policy{3}(3) = 1;
policy{4}(3) = 1;
policy{5}(4) = 1;
policy{6}(4) = 1;
policy{7}(4) = 1;
policy{8}(4) = 1;
policy{9}(5) = 1;
policy{10}(5) = 1;
policy{11}(5) = 1;
policy{12}(5) = 1;
policy{13}(5) = 1;

% Plot NE guess policy
policy_plot_fg = 1;
policy_plot_pos = [0, 2 * default_height, default_width, default_height];
policy_plot_title = 'Current NE Guess Policy';
ne_func.plot_policy(policy_plot_fg, policy_plot_pos, policy, ne_param, policy_plot_title, RedColormap);

% Step 0.2: Stationary distribution at NE policy guess
% Parametrized as a vector with num_k rows. D(i) is probability of k = k(i)
% Note: All entries are non-negative and vector sums to 1
% Initialize uniformly at random
% D = rand(ne_param.num_k, 1);
% D = D / sum(D);
% D = 1 / ne_param.num_k * ones(ne_param.num_k, 1);
% D = zeros(ne_param.num_k, 1);
% D(6) = 1;
% % D(end) = 0.5;
D = [0.127173496578804; 0.189740015866709; 0.109532506434240; 0.136566278445110; 0.149797934910601; 0.0885577991270482; 0.0710842446682071; 0.0454906393703621; 0.0348866305933872; 0.0212892460589850; 0.0147947014226698; 0.00534391059824620; 0.00574259592562984];

%% Step 1: Get (D,T) pair corresponding to NE policy guess
num_iter = 0;

% Step 1.1: Get T for when all agents play policy and stationary
% distribution is D
T = ne_func.get_T(policy, policy, D, ne_param);

% Step 1.2: Get stattionary distribution D from T
D_next = ne_func.get_D(T);

% Step 1.3: Repeat steps 1.1-1.2 until (D,T) pair converge
num_iter_D_T = 0;
D_error = norm(D - D_next, inf);
D = D_next;
while D_error > ne_param.D_T_tol && num_iter_D_T < ne_param.D_T_max_iter
    num_iter_D_T = num_iter_D_T + 1;
    T = ne_func.get_T(policy, policy, D, ne_param);
    D_next = ne_func.get_D(T);
    D_error = norm(D - D_next, inf);
    D = D_next;
end
if num_iter_D_T == ne_param.D_T_max_iter
    fprintf('Iteration %d did not find convergent (D,T) pair\n', num_iter);
end
% Plot stationary distribution
D_plot_fg = 2;
D_plot_pos = [0, default_height / 3, default_width, default_height];
D_plot_title = 'Current NE Stationary Distribution';
ne_func.plot_D(D_plot_fg, D_plot_pos, D, ne_param, D_plot_title);

%% Step 2: Best response of agent i to all other agents j playing policy
% This is a dynamic progarmming problem solved using policy iteration
% Step 2.0: Initialize agent i's best response policy with current NE guess
policy_i = policy;
% Step 2.1: Get current stage cost for agent i playing policy_i
c_i = ne_func.get_c(policy_i, policy, D, ne_param);
% Step 2.2: Get probability transition matrix for agent i playing policy_i
T_i = ne_func.get_T(policy_i, policy, D, ne_param);
% Step 2.3: Get expected utility for agent i playing policy_i
theta_i = (eye(ne_param.num_k) - ne_param.alpha * T_i) \ c_i;
% Plot expected utility
theta_plot_fg = 3;
theta_plot_pos = [default_width, default_height / 3, default_width, default_height];
theta_plot_title = 'Current NE Expected Utility';
ne_func.plot_theta(theta_plot_fg, theta_plot_pos, theta_i, ne_param, theta_plot_title);
% Step 2.4: Get the expected cost matrix for agent i as per the messages
% they would transmit, given current expected utility theta_i
rho_i = ne_func.get_rho(policy, D, theta_i, ne_param);
% Step 2.5: Get next minimizing policy for agent i
policy_i_next = ne_func.get_policy(rho_i);
% Step 2.6: Policy iteration - repeat Steps 2.1-2.5 until policy_i
% converges
policy_i_error = ne_func.policy_norm(policy_i, policy_i_next, inf);
num_policy_iter = 0;
% Display status
fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_iter, num_policy_iter, policy_i_error);
% Initialize next policy iteration
policy_i = policy_i_next;
while policy_i_error > ne_param.policy_tol && num_policy_iter < ne_param.policy_max_iter
    num_policy_iter = num_policy_iter + 1;
    c_i = ne_func.get_c(policy_i, policy, D, ne_param);
    T_i = ne_func.get_T(policy_i, policy, D, ne_param);
    theta_i = (eye(ne_param.num_k) - ne_param.alpha * T_i) \ c_i;
    rho_i = ne_func.get_rho(policy, D, theta_i, ne_param);
    policy_i_next = ne_func.get_policy(rho_i);
    policy_i_error = ne_func.policy_norm(policy_i, policy_i_next, inf);
    % Display status
    fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_iter, num_policy_iter, policy_i_error);
    % Initialize next iteration
    policy_i = policy_i_next;
end
% Plot best response policy
policy_i_plot_fg = 4;
policy_i_plot_pos = [default_width, 2 * default_height, default_width, default_height];
policy_i_plot_title = 'Best Response Policy';
ne_func.plot_policy(policy_i_plot_fg, policy_i_plot_pos, policy_i, ne_param, policy_i_plot_title, RedColormap);

% Set next NE guess to best response, using momentum
policy_next = cell(ne_param.num_k, 1);
for i_k_i = 1 : ne_param.num_k
    policy_next{i_k_i} = (1 - ne_param.tau) * policy{i_k_i} + ne_param.tau * policy_i{i_k_i};
end

%% Step 3: Repeat Step 1-2 until convergence
policy_error = ne_func.policy_norm(policy, policy_next, inf);
% Display status
fprintf('Iteration %d policy error %f\n', num_iter, policy_error);
policy_hist_end = zeros(1, ne_param.num_k);
policy_hist_end(1) = ne_param.k(policy_i{1}==1);
fprintf('Iteration %d policy: %d', num_iter, policy_hist_end(1));
for i_k_i = 2 : ne_param.num_k
    [~, max_i] = max(policy_i{i_k_i});
    policy_hist_end(i_k_i) = ne_param.k(max_i);
    fprintf("->%d", policy_hist_end(i_k_i));
end
policy_hist = policy_hist_end;
fprintf('\n\n');
% Initialize next iteration
policy = policy_next;
% Plot new NE guess policy
ne_func.plot_policy(policy_plot_fg, policy_plot_pos, policy, ne_param, policy_plot_title, RedColormap);
while policy_error > ne_param.policy_tol && num_iter < ne_param.ne_policy_max_iter
    num_iter = num_iter + 1;
    
    % Step 3-1.1: Get T for when all agents play policy and stationary
    % distribution is D
    % Re-initialize D to uniform distribution first
    D = 1 / ne_param.num_k * ones(ne_param.num_k, 1);
    T = ne_func.get_T(policy, policy, D, ne_param);

    % Step 3-1.2: Get stattionary distribution D from T
    D_next = ne_func.get_D(T);

    % Step 3-1.3: Repeat steps 1.1-1.2 until (D,T) pair converge
    num_iter_D_T = 0;
    D_error = norm(D - D_next, inf);
    D = D_next;
    while D_error > ne_param.D_T_tol && num_iter_D_T < ne_param.D_T_max_iter
        num_iter_D_T = num_iter_D_T + 1;
        T = ne_func.get_T(policy, policy, D, ne_param);
        D_next = ne_func.get_D(T);
        D_error = norm(D - D_next, inf);
        D = D_next;
    end
    if num_iter_D_T == ne_param.D_T_max_iter
        fprintf('Iteration %d did not find convergent (D,T) pair\n', num_iter);
    end
    % Plot stationary distribution
    ne_func.plot_D(D_plot_fg, D_plot_pos, D, ne_param, D_plot_title);
    
    % Step 3-2: Best response of agent i to all other agents j playing policy
    % This is a dynamic progarmming problem solved using policy iteration
    % Step 3-2.0: Initialize agent i's best response policy with current NE guess
    policy_i = policy;
    % Step 3-2.1: Get current stage cost for agent i playing policy_i
    c_i = ne_func.get_c(policy_i, policy, D, ne_param);
    % Step 3-2.2: Get probability transition matrix for agent i playing policy_i
    T_i = ne_func.get_T(policy_i, policy, D, ne_param);
    % Step 3-2.3: Get expected utility for agent i playing policy_i
    theta_i = (eye(ne_param.num_k) - ne_param.alpha * T_i) \ c_i;
    % Plot expected utility
    ne_func.plot_theta(theta_plot_fg, theta_plot_pos, theta_i, ne_param, theta_plot_title);
    % Step 3-2.4: Get the expected cost matrix for agent i as per the messages
    % they would transmit, given current expected utility theta_i
    rho_i = ne_func.get_rho(policy, D, theta_i, ne_param);
    % Step 3-2.5: Get next minimizing policy for agent i
    policy_i_next = ne_func.get_policy(rho_i);
    % Step 3-2.6: Policy iteration - repeat Steps 2.1-2.5 until policy_i
    % converges
    policy_i_error = ne_func.policy_norm(policy_i, policy_i_next, inf);
    num_policy_iter = 0;
    % Display status
    fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_iter, num_policy_iter, policy_i_error);
    % Initialize next policy iteration
    policy_i = policy_i_next;
    while policy_i_error > ne_param.policy_tol && num_policy_iter < ne_param.policy_max_iter
        num_policy_iter = num_policy_iter + 1;
        c_i = ne_func.get_c(policy_i, policy, D, ne_param);
        T_i = ne_func.get_T(policy_i, policy, D, ne_param);
        theta_i = (eye(ne_param.num_k) - ne_param.alpha * T_i) \ c_i;
        rho_i = ne_func.get_rho(policy, D, theta_i, ne_param);
        policy_i_next = ne_func.get_policy(rho_i);
        policy_i_error = ne_func.policy_norm(policy_i, policy_i_next, inf);
        % Display status
        fprintf('Iteration %d policy-iteration %d policy_i error %f\n', num_iter, num_policy_iter, policy_i_error);
        % Initialize next policy iteration
        policy_i = policy_i_next;
    end
    % Plot best response policy
    ne_func.plot_policy(policy_i_plot_fg, policy_i_plot_pos, policy_i, ne_param, policy_i_plot_title, RedColormap);
    % Set next NE guess to best response, using momentum
    policy_next = cell(ne_param.num_k, 1);
    for i_k_i = 1 : ne_param.num_k
        policy_next{i_k_i} = (1 - ne_param.tau) * policy{i_k_i} + ne_param.tau * policy_i{i_k_i};
    end
    
    policy_error = ne_func.policy_norm(policy, policy_next, inf);
    % Display status
    fprintf('Iteration %d policy error %f\n', num_iter, policy_error);
    policy_hist_end = zeros(1, ne_param.num_k);
    policy_hist_end(1) = ne_param.k(policy_i{1}==1);
    fprintf('Iteration %d policy: %d', num_iter, policy_hist_end(1));
    for i_k_i = 2 : ne_param.num_k
        [~, max_i] = max(policy_i{i_k_i});
        policy_hist_end(i_k_i) = ne_param.k(max_i);
        fprintf("->%d", policy_hist_end(i_k_i));
    end
    % Detect a limit cycle
    limit_cycle = false;
    for policy_hist_i = 1 : size(policy_hist, 1)4
        if isequal(policy_hist(policy_hist_i,:), policy_hist_end)
            % Limit cycle found
            limit_cycle = true;
            policy_limit_cycle = policy_hist(policy_hist_i:end,:);
            policy_limit_cycle_code = policy_limit_cycle * (1 : ne_param.num_k).';
            break;
        end
    end
    policy_hist = [policy_hist; policy_hist_end];
    fprintf('\n\n');
    if ne_param.tau == 1 && limit_cycle && size(policy_limit_cycle, 1) > 1
        fprintf('Limit cycle found!\n\n');
        break;
    end
    % Initialize next iteration
    policy = policy_next;
    % Plot new NE guess policy
    ne_func.plot_policy(policy_plot_fg, policy_plot_pos, policy, ne_param, policy_plot_title, RedColormap);
end

%% Inform user when done
fprintf('DONE\n\n');