% Gets uniform distribution over karma adjusted to have correct
% average karma k_ave
function s_up_k_uniform = get_sigma_up_k_uniform(k_bar, ne_param)
    i_k_bar = find(ne_param.K == k_bar);
    if k_bar * 2 <= ne_param.k_max
        i_k_bar2 = find(ne_param.K == k_bar * 2);
        s_up_k_uniform = [1 / i_k_bar2 * ones(i_k_bar2, 1); zeros(ne_param.n_k - i_k_bar2, 1)];
    elseif k_bar >= ne_param.k_max
        s_up_k_uniform = zeros(ne_param.n_k, 1);
        s_up_k_uniform(end) = 1;
    else
        s_up_k_uniform = 1 / ne_param.n_k * ones(ne_param.n_k, 1);
        K_small = 0 : k_bar - 1;
        K_big = k_bar + 1 : ne_param.k_max;
        num_K_small = length(K_small);
        num_K_big = length(K_big);
        delta_constant = sum(K_small) / num_K_small - sum(K_big) / num_K_big;
        delta_k_ave = k_bar - ne_param.K.' * s_up_k_uniform;
        delta_p = delta_k_ave / delta_constant;
        s_up_k_uniform(1:i_k_bar-1) = s_up_k_uniform(1:i_k_bar-1) + delta_p / num_K_small;
        s_up_k_uniform(i_k_bar+1:end) = s_up_k_uniform(i_k_bar+1:end) - delta_p / num_K_big;
    end
end