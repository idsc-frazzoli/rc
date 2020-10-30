% Gets stationary distribution of transition matrix T
% This essentially solves d = T.'*d. The solution is the left eigenvector
% corresponding to eigenvalue 1, or the kernel of (I - T.')
function d = stat_dist(T)
    n = size(T, 1);
    left_eig_T_1 = null(eye(n) - T.');
    % Make sure to return a valid probability distribution (sums to 1)
    if ~isempty(left_eig_T_1)
        d = left_eig_T_1 / sum(left_eig_T_1);
    else
        d = 1 / n * ones(n, 1);
    end
end