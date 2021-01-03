% Plot social welfare policy
function plot_sw_pi(fg, position, colormap, sw_pi_down_u_k_up_m, U, K, M, k_ave)
    persistent sw_pi_plot
    num_U = length(U);
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        sw_pi_plot = cell(num_U, 1);
        for i_u = 1 : num_U
            pi_mat = squeeze(sw_pi_down_u_k_up_m(i_u,:,:));
            pi_mat(pi_mat <= 1e-6) = nan;
            subplot(1, num_U, i_u);
            sw_pi_plot{i_u} = heatmap(K, M, pi_mat.', 'ColorbarVisible','off');
            sw_pi_plot{i_u}.YDisplayData = flipud(sw_pi_plot{i_u}.YDisplayData);
            sw_pi_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW policy for u = ', num2str(U(i_u))];
            sw_pi_plot{i_u}.XLabel = 'Karma';
            sw_pi_plot{i_u}.YLabel = 'Message';
            sw_pi_plot{i_u}.FontName = 'Ubuntu';
            sw_pi_plot{i_u}.FontSize = 10;
            if exist('colormap', 'var')
                sw_pi_plot{i_u}.Colormap = colormap;
            end
            sw_pi_plot{i_u}.ColorLimits = [0 1];
            sw_pi_plot{i_u}.CellLabelFormat = '%.2f';
        end
    else
        for i_u = 1 : num_U
            pi_mat = squeeze(sw_pi_down_u_k_up_m(i_u,:,:));
            pi_mat(pi_mat <= 1e-6) = nan;
            sw_pi_plot{i_u}.ColorData = pi_mat.';
            sw_pi_plot{i_u}.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW policy for u = ', num2str(U(i_u))];
        end
    end
end