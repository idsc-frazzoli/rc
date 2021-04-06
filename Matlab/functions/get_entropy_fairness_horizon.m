% Gets entropy of accumulated costs with limited memory
function ent = get_entropy_fairness_horizon(c, t_i, fairness_horizon, param)
    % Get costs accumulated over fairness horizon for all
    % timesteps and all agents
    a = [];
    for i_agent = 1 : param.n_a
        t_i_agent = t_i(i_agent);
        num_hist = max([t_i_agent - fairness_horizon + 1, 1]);
        a_agent = zeros(num_hist, 1);
        for i_hist = num_hist : -1 : 1
            end_i = i_hist + t_i_agent - num_hist;
            start_i = max([end_i - fairness_horizon + 1, 1]);
            a_agent(i_hist) = sum(c(start_i:end_i,i_agent));
        end
        a = [a; a_agent];
    end

    % Get distribution of the accumulated costs
    a_unique = unique(a);
    num_a_unique = length(a_unique);
    a_dist = zeros(num_a_unique, 1);
    for i_a = 1 : num_a_unique
        a_dist(i_a) = length(find(a == a_unique(i_a)));
    end
    a_dist = a_dist / sum(a_dist);

    % Get entropy from distribution
    ent = get_entropy(a_dist);
end