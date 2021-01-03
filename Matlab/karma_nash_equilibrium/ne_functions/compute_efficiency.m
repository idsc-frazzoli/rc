% Computes the efficiency of an interaction coset-stationary
% distribution pair
function e = compute_efficiency(q_down_u_k, d_up_u_k)
    e = -dot(reshape(d_up_u_k, [], 1), reshape(q_down_u_k, [], 1));
end