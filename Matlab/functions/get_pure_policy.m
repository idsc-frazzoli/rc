% Gets pure policy from mixed policy using a threshold
function pi_pure = get_pure_policy(pi, K, param)
    n_k = length(K);
    pi_pure = nan(param.n_mu, param.n_alpha, param.n_u, n_k);
    pi_pure(:,:,:,1) = 0;
    for i_mu = 1 : param.n_mu
        for i_alpha = 1 : param.n_alpha
            for i_u = 1 : param.n_u
                for i_k = 2 : n_k
                    i_max = 1 : i_k;
                    [pi_max, i_pi_max] = max(pi(i_mu,i_alpha,i_u,i_k,i_max));
                    i_max(i_pi_max) = [];
                    pi_max_2 = max(pi(i_mu,i_alpha,i_u,i_k,i_max));
                    if pi_max_2 / pi_max < param.pure_policy_tol
                        pi_pure(i_mu,i_alpha,i_u,i_k) = K(i_pi_max);
                    end
                end
            end
        end
    end
end

