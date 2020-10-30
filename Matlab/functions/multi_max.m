% Checks if there are multiple maximizers and returns one uniformly at
% random
function [v_max, i_max] = multi_max(input)
    % Get maximizers
    [v_max, i_max] = func.multi_maxes(input);
    % Choose one uniformly at random if there are multiple
    num_max = length(i_max);
    if num_max > 1
        i_max = datasample(i_max, 1);
    end
end