% Gets initial policy guess
function pi_down_mu_alpha_u_k_up_b_init = get_pi_init(param, ne_param)
    % Initial policy - independent of agent types
    prob_down_u_k_up_b_init = zeros(param.n_u, ne_param.n_k, ne_param.n_k);
    for i_u = 1 : param.n_u
        for i_k = 1 : ne_param.n_k
            switch ne_param.policy_initialization
                case 0 
                    % Bid urgency
                    i_b = min([i_u, i_k]);
                    prob_down_u_k_up_b_init(i_u,i_k,i_b) = 1;
                case 1
                    % Bid 0.5 * u / u_max * k (~ 'bid half if urgent')
                    b = round(0.5 * param.U(i_u) / param.U(end) * ne_param.K(i_k));
                    i_b = ne_param.K == b;
                    prob_down_u_k_up_b_init(i_u,i_k,i_b) = 1;
                case 2
                    % Bid 1 * u / u_max * k (~ 'bid all if urgent')
                    b = round(ne_param.U(i_u) / ne_param.U(end) * ne_param.K(i_k));
                    i_b = ne_param.K == b;
                    prob_down_u_k_up_b_init(i_u,i_k,i_b) = 1;
                case 3
                    % Bid random
                    prob_down_u_k_up_b_init(i_u,i_k,i_b) = 1 / i_k;
            end
        end
    end
    
    % Duplicate for agent types
    pi_down_mu_alpha_u_k_up_b_init = zeros(param.n_mu, param.n_alpha, param.n_u, ne_param.n_k, ne_param.n_k);
    for i_mu = 1 : param.n_mu
        for i_alpha = 1 : param.n_alpha
            pi_down_mu_alpha_u_k_up_b_init(i_mu,i_alpha,:,:,:) = prob_down_u_k_up_b_init;
        end
    end
end