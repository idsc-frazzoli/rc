% Gets initial population distribution guess
function [prob_down_mu_up_u_init, prob_down_mu_alpha_up_k_init, d_up_mu_alpha_u_k_init] = get_d_init(k_bar, param, ne_param)
    % Initial urgency distribution per urgency type
    prob_down_mu_up_u_init = param.prob_down_mu_up_u;
    
    % Initial karma distribution - independent of agent types
    switch ne_param.karma_initialization
        case 0
            % All agents have average karma k_bar
            sigma_up_k_init = zeros(ne_param.n_k, 1);
            i_k_bar = find(ne_param.K == k_bar);
            sigma_up_k_init(i_k_bar) = 1;
        case 1
            % Uniform distribution over [0 : 2 * k_bar]
            sigma_up_k_init = get_sigma_up_k_uniform(k_bar, ne_param);
    end
    
    % Duplicate for agent types
    prob_down_mu_alpha_up_k_init = zeros(param.n_mu, param.n_alpha, ne_param.n_k);
    for i_mu = 1 : param.n_mu
        for i_alpha = 1 : param.n_alpha
            prob_down_mu_alpha_up_k_init(i_mu,i_alpha,:) = sigma_up_k_init;
        end
    end
    
    % Population distribution is the joint distribution of types and states
    d_up_mu_alpha_u_k_init = zeros(param.n_mu, param.n_alpha, param.n_u, ne_param.n_k);
    for i_mu = 1 : param.n_mu
        for i_alpha = 1 : param.n_alpha
            d_up_mu_alpha_u_k_init(i_mu,i_alpha,:,:) = param.g_up_mu_alpha(i_mu,i_alpha) * outer(prob_down_mu_up_u_init(i_mu,:), prob_down_mu_alpha_up_k_init(i_mu,i_alpha,:));
        end
    end
end