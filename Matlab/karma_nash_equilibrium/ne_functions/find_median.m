% Finds the median of the distribution
function med = find_median(s_up_k, K)
    s_up_k = s_up_k / sum(s_up_k);
    s_up_k_cumsum = cumsum(s_up_k);
    i_med = find(s_up_k_cumsum >= 0.5);
    med = K(i_med(1));
end