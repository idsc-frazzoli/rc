% Gets game tensors
function [zeta_down_u_b_bj, kappa_down_k_b_up_kn_down_bj] = get_game_tensors(param, ne_param)
    % Probability of winning/losing given bids
    gamma_down_b_bj_up_o = zeros(ne_param.n_k, ne_param.n_k, ne_param.n_o);
    for i_b = 1 : ne_param.n_k
        b = ne_param.K(i_b);
        for i_bj = 1 : ne_param.n_k
            bj = ne_param.K(i_bj);
            gamma_down_b_bj_up_o(i_b,i_bj,1) = max([0, min([(b - bj + 1) / 2, 1])]);
            gamma_down_b_bj_up_o(i_b,i_bj,2) = 1 - gamma_down_b_bj_up_o(i_b,i_bj,1);
        end
    end

    % Expected interaction cost tensor
    zeta_down_u_o = outer(param.U, ne_param.O);
    zeta_down_u_b_bj = squeeze(dot2(zeta_down_u_o, permute(gamma_down_b_bj_up_o, [3 1 2]), 2, 1));
    clearvars zeta_down_u_o;

    % Karma Markov chain tensor
    prob_down_k_b_bj_o_up_kn = zeros(ne_param.n_k, ne_param.n_k, ne_param.n_k, ne_param.n_o, ne_param.n_k);
    for i_k = 1 : ne_param.n_k
        k = ne_param.K(i_k);
        for i_b = 1 : i_k
            b = ne_param.K(i_b);
            for i_bj = 1 : ne_param.n_k
                bj = ne_param.K(i_bj);

                switch param.payment_rule
                    case 0      % Pay as bid
                        i_kn_win = find(ne_param.K == k - b);
                        i_kn_lose = find(ne_param.K == min([k + bj, ne_param.k_max]));
                    case 1      % Pay difference
                        i_kn_win = find(ne_param.K == k - (b - bj));
                        i_kn_lose = find(ne_param.K == min([k + (bj - b), ne_param.k_max]));
                end

                if ~isempty(i_kn_win)
                    prob_down_k_b_bj_o_up_kn(i_k,i_b,i_bj,1,i_kn_win) = 1;
                end  
                if ~isempty(i_kn_lose)
                    prob_down_k_b_bj_o_up_kn(i_k,i_b,i_bj,2,i_kn_lose) = 1;
                end
            end
        end
    end
    kappa_down_k_b_up_kn_down_bj = permute(dot2(permute(prob_down_k_b_bj_o_up_kn, [1 5 2 3 4]), gamma_down_b_bj_up_o, 5, 3), [1 3 2 4]);
    clearvars gamma_down_m_mj_up_o phi_down_k_m_mj_o_up_kn;
end
