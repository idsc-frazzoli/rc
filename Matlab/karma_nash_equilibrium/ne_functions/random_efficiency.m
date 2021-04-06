% Efficiency of baseline random
function e_rand = random_efficiency(ne_param)
    nu_up_u = func.stat_dist(ne_param.mu_down_u_up_un);
    e_rand = -0.5 * dot(nu_up_u, ne_param.U);
end