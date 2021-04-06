% Infinite horizon payoff of baseline random
function J_rand = random_inf_horizon_payoff(ne_param, alpha)
    if alpha == 1
        J_rand = ne_func.random_efficiency(ne_param) * ones(ne_param.num_U, 1);
    else
        Q = -0.5 * ne_param.U;
        J_rand = (eye(ne_param.num_U) - alpha * ne_param.mu_down_u_up_un) \ Q;
    end
end