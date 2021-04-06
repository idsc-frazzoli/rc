% Gets the entropy of the given distribution
function ent = get_entropy(dist)
    dist(dist == 0) = [];
    ent = -sum(dist .* log2(dist));
end