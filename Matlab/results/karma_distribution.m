clear;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);

%% Some parameters
file_str = 'results/N_200_T_100000_U_1_2_p_0.50_m_0_no_sat/k_ave_';

% Flag to save plots
save_plots = true;

%% Select k_ave(s) and alpha(s) to process
while true
    k_ave_vec = input('Enter k_ave(s): ');
    alpha_vec = input('Enter alpha(s): ');
    
    for i_k_ave = 1 : length(k_ave_vec)
        k_ave = k_ave_vec(i_k_ave);
        
        for i_alpha = 1 : length(alpha_vec)
            alpha = alpha_vec(i_alpha);
            file = [file_str, num2str(k_ave, '%02d'), '_alpha_', num2str(abs(alpha), '%.2f'), '.mat'];
            if ~exist(file, 'file')
                file = [file_str, num2str(k_ave, '%02d'), '.mat'];
            end
            if alpha == -1
                load(file, 'k_bid_u', 'param');
                k = k_bid_u;
            else
                load(file, 'k_ne', 'K_ne', 'param');
                i_alpha_sim = find(param.alpha == alpha);
                if isempty(i_alpha_sim)
                    continue;
                end
                k = k_ne{i_alpha_sim};
            end
            
            k_min = min(k(:));
            k_max = max(k(:));
            K = k_min : k_max;
            num_K = length(K);
            k_dist = zeros(param.tot_num_inter, length(K));
            for t = 1 : param.tot_num_inter
                for i_k = 1 : num_K
                    k_dist(t,i_k) = length(find(k(t,:) == K(i_k))) / param.N;
                end
            end

            % Plot
            close all;
            karma_dist_plot_fg = 5;
            karma_dist_plot_pos = [0, screenheight / 2, screenwidth / 2, screenheight / 2];
            func.plot_karma_dist(karma_dist_plot_fg, karma_dist_plot_pos, k_dist, K, k_ave, alpha);
            drawnow;
            
            if save_plots
                if ~exist('results/karma_distributions', 'dir')
                    mkdir('results/karma_distributions');
                end
                saveas(karma_dist_plot_fg, ['results/karma_distributions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.png']);
                save(['results/karma_distributions/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.mat'], 'k_dist');
            end
        end
    end
end