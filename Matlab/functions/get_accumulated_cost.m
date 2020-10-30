% Gets accumulated costs
function a = get_accumulated_cost(c, t_i, param)
    a = nan(max(t_i), param.n_a);
    for i_agent = 1 : param.n_a
        a(1:t_i(i_agent),i_agent) = cumsum(c(1:t_i(i_agent),i_agent));
        % Normalize by number of interactions
        a(1:t_i(i_agent),i_agent) = a(1:t_i(i_agent),i_agent) ./ (1 : t_i(i_agent)).';
        % Fill the end of the matrix up with the last total
        % accumulated cost
        a(t_i(i_agent)+1:end,i_agent) = a(t_i(i_agent),i_agent);
    end
end