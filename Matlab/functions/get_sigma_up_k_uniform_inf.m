% Gets uniform distribution over karma adjusted to have correct
% average karma k_bar for an infinite population
function [sigma_up_k_uniform, K_uniform] = get_sigma_up_k_uniform_inf(k_bar)
    K_uniform = (0 : k_bar * 2).';
    n_k = length(K_uniform);
    sigma_up_k_uniform = 1 / n_k * ones(n_k, 1);
end