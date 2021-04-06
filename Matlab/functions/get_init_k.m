% Gets a karma initialization for finite n_a agents that is close
% in distribution to specified infinite distribution and has the
% correct k_bar/k_tot
function init_k = get_init_k(sigma_up_k_inf, K, param)
    % Get finite n_a distribution close to infinite distribution
    % and with correct k_bar/k_tot
    sigma_up_k_n_a = round(param.n_a * sigma_up_k_inf);
    sigma_up_k_n_a = reshape(sigma_up_k_n_a, 1, []);
    missing_agents = param.n_a - sum(sigma_up_k_n_a);
    missing_karma = param.k_tot - sigma_up_k_n_a * K;
    while missing_agents ~= 0 || missing_karma ~= 0
        if missing_agents ~= 0
            % Need to adjust agent count (and possibly karma)
            karma_to_adjust = min([floor(missing_karma / missing_agents), K(end)]);
            if karma_to_adjust >= 0
                % Need to either add both agents and karma or
                % remove both agents and karma
                i_karma_to_adjust = find(K == karma_to_adjust);
                agents_to_adjust = missing_agents - rem(missing_karma, missing_agents);
                if agents_to_adjust < 0 && sigma_up_k_n_a(i_karma_to_adjust) == 0
                    % Need to remove agents from a karma that
                    % doesn't have agents. Find closest karma with
                    % agents
                    karma_with_agents = K(sigma_up_k_n_a > 0);
                    [~, i_closest_karma_with_agents] = min(abs(karma_with_agents - K(i_karma_to_adjust)));
                    i_karma_to_adjust = find(K == karma_with_agents(i_closest_karma_with_agents));
                end
                sigma_up_k_n_a(i_karma_to_adjust) = max([sigma_up_k_n_a(i_karma_to_adjust) + agents_to_adjust, 0]);
            else
                if missing_agents > 0
                    % Need to add agents and remove karma
                    % First remove one agent with the closest
                    % amount of karma to what needs to be removed
                    i_karma_to_remove = find(K == min([abs(missing_karma), K(end)]));
                    if sigma_up_k_n_a(i_karma_to_remove) == 0
                        karma_with_agents = K(sigma_up_k_n_a > 0);
                        [~, i_closest_karma_with_agents] = min(abs(karma_with_agents - K(i_karma_to_remove)));
                        i_karma_to_remove = find(K == karma_with_agents(i_closest_karma_with_agents));
                    end
                    sigma_up_k_n_a(i_karma_to_remove) = sigma_up_k_n_a(i_karma_to_remove) - 1;
                    % Now add the required amount of agents with
                    % zero karma to not change karma count
                    sigma_up_k_n_a(1) = sigma_up_k_n_a(1) + missing_agents + 1;
                else
                    % Need to remove agents and add karma
                    % First remove agents with least amount of
                    % karma, and keep track of amount of karma
                    % removed in the process. Remove one extra
                    % agent because one will be added with the
                    % required amount of karma
                    agents_to_remove = abs(missing_agents) + 1;
                    agents_removed = 0;
                    karma_removed = 0;
                    i_karma_to_remove = 1;
                    while agents_removed < agents_to_remove && i_karma_to_remove <= length(K)
                        agents_can_remove = min(agents_to_remove - agents_removed, sigma_up_k_n_a(i_karma_to_remove));
                        sigma_up_k_n_a(i_karma_to_remove) = sigma_up_k_n_a(i_karma_to_remove)- agents_can_remove;
                        agents_removed = agents_removed + agents_can_remove;
                        karma_removed = karma_removed + agents_can_remove * K(i_karma_to_remove);
                        i_karma_to_remove = i_karma_to_remove + 1;
                    end
                    % Now add one agent with the required amount of
                    % karma
                    i_karma_to_add = find(K == min([missing_karma + karma_removed, K(end)]));
                    sigma_up_k_n_a(i_karma_to_add) = sigma_up_k_n_a(i_karma_to_add) + 1;
                end
            end
        else
            if missing_karma > 0
                % Need to add karma only
                % Remove one agent with least karma and add one
                % with the required amount of karma
                karma_with_agents = K(sigma_up_k_n_a > 0);
                i_karma_to_remove = find(K == karma_with_agents(1));
                sigma_up_k_n_a(i_karma_to_remove) = sigma_up_k_n_a(i_karma_to_remove) - 1;
                i_karma_to_add = find(K == min([missing_karma + K(i_karma_to_remove), K(end)]));
                sigma_up_k_n_a(i_karma_to_add) = sigma_up_k_n_a(i_karma_to_add) + 1;
            else
                % Need to remove karma only
                % Remove one agent with the closest amount of karma
                % to what needs to be removed and add one agent
                % with zero karma
                i_karma_to_remove = find(K == min([abs(missing_karma), K(end)]));
                if sigma_up_k_n_a(i_karma_to_remove) == 0
                    karma_with_agents = K(sigma_up_k_n_a > 0);
                    [~, i_closest_karma_with_agents] = min(abs(karma_with_agents - K(i_karma_to_remove)));
                    i_karma_to_remove = find(K == karma_with_agents(i_closest_karma_with_agents));
                end
                sigma_up_k_n_a(i_karma_to_remove) = sigma_up_k_n_a(i_karma_to_remove) - 1;
                sigma_up_k_n_a(1) = sigma_up_k_n_a(1) + 1;
            end
        end
        missing_agents = param.n_a - sum(sigma_up_k_n_a);
        missing_karma = param.k_tot - sigma_up_k_n_a * K;
    end

    % Initialize karma for N agents as per finite n_a distribution
    init_k = zeros(1, param.n_a);
    start_i = 0;
    for i_k = 1 : length(K)
        num_agents = sigma_up_k_n_a(i_k);
        init_k(start_i+1:start_i+num_agents) = K(i_k);
        start_i = start_i + num_agents;
    end
end