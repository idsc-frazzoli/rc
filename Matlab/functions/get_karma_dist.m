% Gets the empirical karma distribution for the whole society and
% per agent
function [k_dist, k_dist_agents] = get_karma_dist(k, param)
    k_min = nanmin(k(:));
    k_max = nanmax(k(:));
    K = k_min : k_max;
    n_K = length(K);
    k_dist = zeros(n_K, 1);
    k_dist_agents = zeros(n_K, param.n_a);
    for i_k = 1 : n_K
        k_dist(i_k) = length(find(k(:) == K(i_k)));
        for i_agent = 1 : param.n_a
            k_dist_agents(i_k,i_agent) = length(find(k(:,i_agent) == K(i_k)));
        end
    end
    k_dist = k_dist / sum(k_dist);
    k_dist_agents = k_dist_agents ./ sum(k_dist_agents, 1);
end