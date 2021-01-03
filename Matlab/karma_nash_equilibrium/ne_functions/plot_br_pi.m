% Plot best response policy
function plot_br_pi(fg, position, colormap, br_pi_down_mu_alpha_u_k_up_b, param, ne_param, i_alpha_comp)
    persistent br_pi_plot
    if ~ishandle(fg)
        figure(fg);
        fig = gcf;
        fig.Position = position;
        br_pi_plot = cell(param.n_mu, param.n_alpha, param.n_u);
        n_subplots = param.n_mu * param.n_alpha * param.n_u;
        n_cols = ceil(sqrt(n_subplots));
        n_rows = ceil(n_subplots / n_cols);
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                for i_u = 1 : param.n_u
                    pi_mat = squeeze(br_pi_down_mu_alpha_u_k_up_b(i_mu,i_alpha,i_u,:,:));
                    pi_mat(pi_mat <= eps) = nan;
                    i_subplot = (i_mu - 1) * (param.n_alpha * param.n_u) + (i_alpha - 1) * param.n_u + i_u;
                    subplot(n_rows, n_cols, i_subplot);
                    br_pi_plot{i_mu,i_alpha,i_u} = heatmap(ne_param.K, ne_param.K, pi_mat.', 'ColorbarVisible','off');
                    br_pi_plot{i_mu,i_alpha,i_u}.YDisplayData = flipud(br_pi_plot{i_mu,i_alpha,i_u}.YDisplayData);
                    alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                    if alpha > 0.99 && alpha < 1
                        alpha_str = num2str(alpha, '%.3f');
                    else
                        alpha_str = num2str(alpha, '%.2f');
                    end
                    br_pi_plot{i_mu,i_alpha,i_u}.Title = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' BR policy for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str, ', u = ', num2str(param.U(i_u))];
                    br_pi_plot{i_mu,i_alpha,i_u}.XLabel = 'Karma';
                    br_pi_plot{i_mu,i_alpha,i_u}.YLabel = 'Bid';
                    br_pi_plot{i_mu,i_alpha,i_u}.FontName = 'Ubuntu';
                    br_pi_plot{i_mu,i_alpha,i_u}.FontSize = 10;
                    if exist('colormap', 'var')
                        br_pi_plot{i_mu,i_alpha,i_u}.Colormap = colormap;
                    end
                    br_pi_plot{i_mu,i_alpha,i_u}.ColorLimits = [0 1];
                    br_pi_plot{i_mu,i_alpha,i_u}.CellLabelFormat = '%.2f';
                end
            end
        end
    else
        for i_mu = 1 : param.n_mu
            for i_alpha = 1 : param.n_alpha
                for i_u = 1 : param.n_u
                    pi_mat = squeeze(br_pi_down_mu_alpha_u_k_up_b(i_mu,i_alpha,i_u,:,:));
                    pi_mat(pi_mat <= eps) = nan;
                    br_pi_plot{i_mu,i_alpha,i_u}.ColorData = pi_mat.';
                    alpha = param.Alpha(max([i_alpha, i_alpha_comp]));
                    if alpha > 0.99 && alpha < 1
                        alpha_str = num2str(alpha, '%.3f');
                    else
                        alpha_str = num2str(alpha, '%.2f');
                    end
                    br_pi_plot{i_mu,i_alpha,i_u}.Title = ['k_{bar} = ', num2str(param.k_bar, '%02d'), ' BR policy for \mu = ', num2str(i_mu), ', \alpha = ', alpha_str, ', u = ', num2str(param.U(i_u))];
                end
            end
        end
    end
end