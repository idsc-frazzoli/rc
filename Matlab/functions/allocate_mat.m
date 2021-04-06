% Allocates matrix to store agent interaction data
% Row => Timestep
% Col => Agent
% Values are initialized with nan to tell last timestep agents
% participated in an interaction
function mat = allocate_mat(param)
    mat = nan(param.max_T_i, param.n_a);
end