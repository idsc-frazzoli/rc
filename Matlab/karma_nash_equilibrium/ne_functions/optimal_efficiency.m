% Computes the optimal efficiency (which corresponds to centralized
% urgency)
function e_opt = optimal_efficiency(ne_param)
    nu_up_u = func.stat_dist(ne_param.mu_down_u_up_un);
    e_opt = 0;
    for i_ui = 1 : ne_param.num_U
        e_opt = e_opt - 0.5 * nu_up_u(i_ui)^2 * ne_param.U(i_ui);
        for i_uj = i_ui + 1 : ne_param.num_U
            e_opt = e_opt - nu_up_u(i_ui) * nu_up_u(i_uj) * ne_param.U(i_ui);
        end
    end
end