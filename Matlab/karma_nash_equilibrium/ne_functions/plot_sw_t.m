% Plot SW state transitions
function plot_sw_t(fg, position, colormap, sw_t_down_u_k_up_un_kn, U, K, k_ave)
    persistent sw_t_plot
    num_U = length(U);
    num_K = length(K);
    num_X = num_U * num_K;
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        label = cell(num_X, 1);
        for i_u = 1 : num_U
            base_i_u = (i_u - 1) * num_K;
            u_str = num2str(U(i_u));
            for i_k = 1 : num_K
                label{base_i_u+i_k} = ['(', u_str, ',', num2str(K(i_k)), ')'];
            end
        end
        t_mat = zeros(num_X);
        for i_u = 1 : num_U
            start_i_u = (i_u - 1) * num_K + 1;
            end_i_u = i_u * num_K;
            for i_un = 1 : num_U
                start_i_un = (i_un - 1) * num_K + 1;
                end_i_un = i_un * num_K;
                t_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                    squeeze(sw_t_down_u_k_up_un_kn(i_u,:,i_un,:));
            end
        end
        sw_t_plot = heatmap(label, label, t_mat.', 'ColorbarVisible','off');
        sw_t_plot.YDisplayData = flipud(sw_t_plot.YDisplayData);
        sw_t_plot.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW state transitions'];
        sw_t_plot.XLabel = 'State now (urgency,karma)';
        sw_t_plot.YLabel = 'State next (urgency,karma)';
        sw_t_plot.FontName = 'Ubuntu';
        sw_t_plot.FontSize = 10;
        if exist('colormap', 'var')
            sw_t_plot.Colormap = colormap;
        end
        sw_t_plot.CellLabelFormat = '%.2f';
    else
        t_mat = zeros(num_X);
        for i_u = 1 : num_U
            start_i_u = (i_u - 1) * num_K + 1;
            end_i_u = i_u * num_K;
            for i_un = 1 : num_U
                start_i_un = (i_un - 1) * num_K + 1;
                end_i_un = i_un * num_K;
                t_mat(start_i_u:end_i_u,start_i_un:end_i_un) =...
                    squeeze(sw_t_down_u_k_up_un_kn(i_u,:,i_un,:));
            end
        end
        sw_t_plot.ColorData = t_mat.';
        sw_t_plot.Title = ['k_{avg} = ', num2str(k_ave, '%02d'), ' SW state transitions'];
    end
end