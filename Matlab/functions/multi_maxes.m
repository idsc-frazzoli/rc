% Returns all maximizers (if there are multiple)
function [v_max, i_max] = multi_maxes(input)
    [v_max, i_max] = max(input);
    input(i_max) = -realmax;
    [next_v_max, next_i_max] = max(input);
    while next_v_max == v_max
        i_max = [i_max, next_i_max];
        input(next_i_max) = -realmax;
        [next_v_max, next_i_max] = max(input);
    end
end