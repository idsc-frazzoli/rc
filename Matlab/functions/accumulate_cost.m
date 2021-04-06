% Accumulates the cost in the history up to specified fairness
% horizon. Fairness horizon = inf means all the history
function a = accumulate_cost(c, agents_id, u, t_i, fairness_horizon)
    a = zeros(1, 2);
    for i_agent = 1 : 2
        id = agents_id(i_agent);
        if fairness_horizon == inf
            start_i = 1;
        else
            start_i = max([t_i(id) - fairness_horizon, 1]);
        end
        a(i_agent) = sum(c(start_i:t_i(id)-1,id)) + u(i_agent);
        a(i_agent) = a(i_agent) / min([fairness_horizon + 1, t_i(id)]);
    end
end