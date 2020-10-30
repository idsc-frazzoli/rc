% Allocates karma matrix and sets the initial karma as per init_k
function k = allocate_karma(param, init_k)
    k = allocate_mat(param);
    k(1,:) = init_k;
end