% Sets axis limit 'scale' above/below min-max values
function axis_semi_tight(ax, scale)
    axis tight; % Set axis tight
    % x limits
    xl = xlim(ax); % Get tight axis limits
    range = xl(2) - xl(1); % Get tight axis range
    sc_range = scale * range; % Scale range
    xl(1) = xl(1) - (sc_range - range) / 2; % New xmin
    xl(2) = xl(1) + sc_range; % New xmax
    xlim(ax, xl);
    % y limits
    yl = ylim(ax); % Get tight axis limits
    range = yl(2) - yl(1); % Get tight axis range
    sc_range = scale * range; % Scale range
    yl(1) = yl(1) - (sc_range - range) / 2; % New ymin
    yl(2) = yl(1) + sc_range; % New ymax
    ylim(ax, yl);
end