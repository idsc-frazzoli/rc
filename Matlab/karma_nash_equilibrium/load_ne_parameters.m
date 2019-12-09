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

% Vector of all outcome values
ne_param.O = [0; 1];

% Number of outcome values
ne_param.num_O = length(ne_param.O);

% k_next cell of matrices. Each pair of 2 agents' (k_i, k_j) has a matrix
% (i,j). Each matrix describes next karma for agent i as a function of
% their bid m_i (rows) and other agent j's bid m_j (cols). Note that agents
% are limited in their bids by their karma level, which is why matrices
% have different dimensions
ne_param.k_next = cell(ne_param.num_K);
for i_k_i = 1 : ne_param.num_K
    k_i = ne_param.K(i_k_i);
    for i_k_j = 1 : ne_param.num_K
        k_j = ne_param.K(i_k_j);
        ne_param.k_next{i_k_i,i_k_j} = cell(i_k_i, i_k_j);
        for i_m_i = 1 : i_k_i
            m_i = ne_param.K(i_m_i);
            for i_m_j = 1 : i_k_j
                m_j = ne_param.K(i_m_j);
                
                % Next karma level if agent i is to receive karma
                k_in = min([k_i + m_j, param.k_max]);
                % Next karma level if agent i is to pay karma
                k_out = k_i - min([m_i, param.k_max - k_j]);
                
                % Agent i receives karma when they bid lower than agent j
                if m_i < m_j
                    ne_param.k_next{i_k_i,i_k_j}{i_m_i,i_m_j} = k_in;
                % Agent i pays karma when they bid higher than agent j
                elseif m_i > m_j
                    ne_param.k_next{i_k_i,i_k_j}{i_m_i,i_m_j} = k_out;
                % Agent i can either pay or receive karma on equal bids
                % (50/50 chances). We keep track of both options here
                else
                    ne_param.k_next{i_k_i,i_k_j}{i_m_i,i_m_j} = [k_in, k_out];
                end
            end
        end
    end
end

% Number of states, which is number of urgency * number of karma values
ne_param.num_X = ne_param.num_U * ne_param.num_K;

% Alpha
ne_param.alpha = 0.8;

% Tolerance for convergence of (D,T) pair
ne_param.D_T_tol = 1e-4;

% Maximum number of iterations for convergence of (D,T) pair
ne_param.D_T_max_iter = 100;

% Tolerance for convergence of D
ne_param.D_tol = 1e-4;

% Maximum number of iterations for convergence of D
ne_param.D_max_iter = 1000;

% Tolerance for convergence of V
ne_param.V_tol = 1e-4;

% Maximum number of iterations for convergence of V
ne_param.V_max_iter = 1000;

% Momentum
ne_param.tau = 1.0;

% Tolerance for convergence of policy
ne_param.policy_tol = 1e-3;

% Maximum number of policy iterations
ne_param.policy_max_iter = 100;

% Maximum number of Nash Equilibrium policy iterations
ne_param.ne_policy_max_iter = 1000;

% Do plots
ne_param.plot = true;

end