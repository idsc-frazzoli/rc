clear;
close all;
clc;

%% Screen size used to place plots
screensize = get(groot, 'ScreenSize');
screenwidth = screensize(3);
screenheight = screensize(4);
default_width = screenwidth / 2;
default_height = screenheight / 2;
load('karma_nash_equilibrium/RedColormap.mat');

%% Parameters
% NE computation parameters
ne_param = load_ne_parameters();

%% Loop over k_ave(s)
for i_k_ave = 1 : length(ne_param.k_ave)
    k_ave = ne_param.k_ave(i_k_ave);
    %% Loop over alpha(s)
    for i_alpha = 1 : length(ne_param.alpha)
        alpha = ne_param.alpha(i_alpha);
        fprintf('%%%%K_AVE = %d ALPHA = %f%%%%\n\n', k_ave, alpha);
                
        %% Guess an initial distribution of urgency and karma
        % Set initial distribution of urgency to stationary distribution of urgency
        % markov chain 
        p_up_u = ne_func.stat_dist(ne_param.mu_down_u_up_un);

        % Set initial distribution of karma to be uniform around k_ave
        k_max = k_ave * 2;
        K = (0 : k_max).';
        num_K = length(K);
        s_up_k = 1 / num_K * ones(num_K, 1);

        %% Guess policy for initial set of K
        m_down_u_k = zeros(ne_param.num_U, num_K);
        for i_u = 1 : ne_param.num_U
            if ne_param.U(i_u) ~= 0
                m_down_u_k(i_u,:) = round(0.5 * ne_param.U(i_u) / ne_param.U(end) * K.');
            end
        end

        %% Get probability distribution over seeing messages
        M = unique(m_down_u_k);
        num_M = length(M);
        iota_up_m = zeros(num_M, 1);
        for i_u = 1 : ne_param.num_U
            if ne_param.U(i_u) == 0
                iota_up_m(1) = iota_up_m(1) + p_up_u(i_u);
            else
                for i_m = 1 : num_M
                    iota_up_m(i_m) = iota_up_m(i_m) + p_up_u(i_u) * sum(s_up_k(m_down_u_k(i_u,:) == M(i_m)));
                end
            end
        end

        %% Get karma transition matrices for non-urgent and urgent agents
        % Matrices are 'fat'. We consider only next karma values that are possible
        kn_max = k_max + M(end);
        num_Kn = kn_max + 1;
        theta_down_u_k_up_kn = zeros(ne_param.num_U, num_K, num_Kn);
        for i_u = 1 : ne_param.num_U
            for i_k = 1 : num_K
                m = m_down_u_k(i_u,i_k);
                if m == 0
                    theta_down_u_k_up_kn(i_u,i_k,i_k+M) = iota_up_m.';
                else
                    i_m_win = find(M < m);
                    i_kn_win = i_k - m;
                    theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = sum(iota_up_m(i_m_win));
                    i_m_tie = find(M == m);
                    if ~isempty(i_m_tie)
                        i_kn_tie = i_k + m;
                        theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = theta_down_u_k_up_kn(i_u,i_k,i_kn_win) + 0.5 * iota_up_m(i_m_tie);
                        theta_down_u_k_up_kn(i_u,i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
                    end
                    i_m_lose = find(M > m);
                    if ~isempty(i_m_lose)
                        i_kn_lose = i_k + M(i_m_lose);
                        theta_down_u_k_up_kn(i_u,i_k,i_kn_lose) = iota_up_m(i_m_lose).';
                    end
                end
            end

        end
        t_down_k_up_kn = zeros(num_K, num_Kn);
        for i_u = 1 : ne_param.num_U
            t_down_k_up_kn = t_down_k_up_kn + p_up_u(i_u) * squeeze(theta_down_u_k_up_kn(i_u,:,:));
        end

        %% Now take a step in the direction of transition matrix
        s_up_k_next = t_down_k_up_kn.' * s_up_k;

        %% Now pluck out unlikely states form the end
        while s_up_k_next(end) < 1e-8
            s_up_k_next(end) = [];
        end

        %% Renormalize
        s_up_k_next = s_up_k_next / sum(s_up_k_next);

        %% Repeat all of the above until convegrence
        if length(s_up_k) == length(s_up_k_next)
            s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
        else
            s_converged = false;
        end
        s_up_k = s_up_k_next;
        while ~s_converged
            k_max = length(s_up_k) - 1;
            K = (0 : k_max).';
            num_K_prev = num_K;
            num_K = length(K);

            %% Get probability distribution over seeing messages
            if num_K > num_K_prev
                m_down_u_k = [m_down_u_k, repmat(m_down_u_k(:,end), 1, num_K - num_K_prev)];
            elseif num_K < num_K_prev
                m_down_u_k(:,num_K+1:end) = [];
            end
            M = unique(m_down_u_k);
            num_M = length(M);
            iota_up_m = zeros(num_M, 1);
            for i_u = 1 : ne_param.num_U
                if ne_param.U(i_u) == 0
                    iota_up_m(1) = iota_up_m(1) + p_up_u(i_u);
                else
                    for i_m = 1 : num_M
                        iota_up_m(i_m) = iota_up_m(i_m) + p_up_u(i_u) * sum(s_up_k(m_down_u_k(i_u,:) == M(i_m)));
                    end
                end
            end

            %% Get karma transition matrices for non-urgent and urgent agents
            % Matrices are 'fat'. We consider only next karma values that are possible
            kn_max = k_max + M(end);
            num_Kn = kn_max + 1;
            theta_down_u_k_up_kn = zeros(ne_param.num_U, num_K, num_Kn);
            for i_u = 1 : ne_param.num_U
                for i_k = 1 : num_K
                    m = m_down_u_k(i_u,i_k);
                    if m == 0
                        theta_down_u_k_up_kn(i_u,i_k,i_k+M) = iota_up_m.';
                    else
                        i_m_win = find(M < m);
                        i_kn_win = i_k - m;
                        theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = sum(iota_up_m(i_m_win));
                        i_m_tie = find(M == m);
                        if ~isempty(i_m_tie)
                            i_kn_tie = i_k + m;
                            theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = theta_down_u_k_up_kn(i_u,i_k,i_kn_win) + 0.5 * iota_up_m(i_m_tie);
                            theta_down_u_k_up_kn(i_u,i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
                        end
                        i_m_lose = find(M > m);
                        if ~isempty(i_m_lose)
                            i_kn_lose = i_k + M(i_m_lose);
                            theta_down_u_k_up_kn(i_u,i_k,i_kn_lose) = iota_up_m(i_m_lose).';
                        end
                    end
                end

            end
            t_down_k_up_kn = zeros(num_K, num_Kn);
            for i_u = 1 : ne_param.num_U
                t_down_k_up_kn = t_down_k_up_kn + p_up_u(i_u) * squeeze(theta_down_u_k_up_kn(i_u,:,:));
            end

            %% Now take a step in the direction of transition matrix
            s_up_k_next = t_down_k_up_kn.' * s_up_k;

            %% Now pluck out unlikely states form the end
            while s_up_k_next(end) < 1e-8
                s_up_k_next(end) = [];
            end

            %% Renormalize
            s_up_k_next = s_up_k_next / sum(s_up_k_next);

            %% Repeat all of the above until convegrence
            if length(s_up_k) == length(s_up_k_next)
                s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
            else
                s_converged = false;
            end
            s_up_k = s_up_k_next;
        end

        %% Get expected stage cost
        q_down_u_k = zeros(ne_param.num_U, num_K);
        for i_u = 1 : ne_param.num_U
            if ne_param.U(i_u) == 0
                continue;
            end
            for i_k = 1 : num_K
                m = m_down_u_k(i_u,i_k);
                i_m_tie = find(M == m);
                if ~isempty(i_m_tie)
                    q_down_u_k(i_u,i_k) = 0.5 * iota_up_m(i_m_tie) * ne_param.U(i_u);
                end
                i_m_lose = find(M > m);
                q_down_u_k(i_u,i_k) = q_down_u_k(i_u,i_k) + sum(iota_up_m(i_m_lose)) * ne_param.U(i_u);
            end
        end

        %% Concatenate
        % Get rid of 'extra' states. Is this the same as saturation?
        theta_down_u_k_up_kn_trunc = theta_down_u_k_up_kn(:,:,1:num_K);
        theta_down_u_k_up_kn_trunc(:,:,end) = 1 - sum(theta_down_u_k_up_kn_trunc(:,:,1:end-1), 3);
        t_down_u_k_up_un_kn = zeros(ne_param.num_U, num_K, ne_param.num_U, num_K);
        for i_u = 1 : ne_param.num_U
            for i_un = 1 : ne_param.num_U
                t_down_u_k_up_un_kn(i_u,:,i_un,:) = ne_param.mu_down_u_up_un(i_u,i_un) * squeeze(theta_down_u_k_up_kn_trunc(i_u,:,:));
            end
        end

        %% Get expected infinite horizon cost
        v_down_u_k = zeros(ne_param.num_U, num_K);
        v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
        num_v_iter = 1;
        v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
        v_down_u_k = v_down_u_k_next;
        while ~v_converged
            v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
            num_v_iter = num_v_iter + 1;
            v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
            v_down_u_k = v_down_u_k_next;
        end

        %% Get best response
        % Only do so for 'prevalent' k's
        s_up_k_trunc = s_up_k;
        while s_up_k_trunc(end) < 1e-3
            s_up_k_trunc(end) = [];
        end
        num_K_trunc = length(s_up_k_trunc);
        br_m_down_u_k = m_down_u_k(:,1:num_K_trunc);
        for i_u = 1 : ne_param.num_U
            if ne_param.U(i_u) == 0
                continue;
            end
            for i_k = 1 : num_K_trunc
                if K(i_k) == 0
                    continue;
                end
                rho_down_u_k_m_min = v_down_u_k(i_u,i_k);
                for m = br_m_down_u_k(i_u,i_k-1) : K(i_k)
                    if m == m_down_u_k(i_u,i_k)
                        continue;
                    end
                    % Expected stage cost for current message
                    i_m_tie = find(M == m);
                    if isempty(i_m_tie)
                        eta_down_u_k_m = 0;
                    else
                        eta_down_u_k_m = 0.5 * iota_up_m(i_m_tie) * ne_param.U(i_u);
                    end
                    i_m_lose = find(M > m);
                    eta_down_u_k_m = eta_down_u_k_m + sum(iota_up_m(i_m_lose)) * ne_param.U(i_u);
                    % Expected transition probabilities for current message
                    tau_down_u_k_m_up_kn = zeros(1, num_Kn);
                    if m == 0
                        tau_down_u_k_m_up_kn(i_k+M) = iota_up_m.';
                    else
                        i_m_win = find(M < m);
                        i_kn_win = i_k - m;
                        tau_down_u_k_m_up_kn(i_kn_win) = sum(iota_up_m(i_m_win));
                        if ~isempty(i_m_tie)
                            i_kn_tie = i_k + m;
                            tau_down_u_k_m_up_kn(i_kn_win) = tau_down_u_k_m_up_kn(i_kn_win) + 0.5 * iota_up_m(i_m_tie);
                            tau_down_u_k_m_up_kn(i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
                        end
                        if ~isempty(i_m_lose)
                            i_kn_lose = i_k + M(i_m_lose);
                            tau_down_u_k_m_up_kn(i_kn_lose) = iota_up_m(i_m_lose).';
                        end
                    end

                    tau_down_u_k_m_up_kn_trunc = tau_down_u_k_m_up_kn(1:num_K);
                    tau_down_u_k_m_up_kn_trunc(end) = 1 - sum(tau_down_u_k_m_up_kn_trunc(1:end-1));
                    chi_down_u_k_m_up_un_kn = zeros(ne_param.num_U, num_K);
                    for i_un = 1 : ne_param.num_U
                        chi_down_u_k_m_up_un_kn(i_un,:) = ne_param.mu_down_u_up_un(i_u,i_un) * tau_down_u_k_m_up_kn_trunc;
                    end
                    rho_down_u_k_m = eta_down_u_k_m + alpha * reshape(chi_down_u_k_m_up_un_kn, [], 1).' * reshape(v_down_u_k, [], 1);
                    if rho_down_u_k_m < rho_down_u_k_m_min - ne_param.br_v_tol
                        br_m_down_u_k(i_u,i_k) = m;
                        rho_down_u_k_m_min = rho_down_u_k_m;
                    end
                end
                if br_m_down_u_k(i_u,i_k) < br_m_down_u_k(i_u,i_k-1)
                    br_m_down_u_k(i_u,i_k) = br_m_down_u_k(i_u,i_k-1);
                end
            end
        end

        %% Repeat until convergence
        ne_converged = norm(m_down_u_k(:,1:num_K_trunc) - br_m_down_u_k, inf) <= ne_param.br_pi_tol;
        m_down_u_k = [br_m_down_u_k, repmat(br_m_down_u_k(:,end), 1, num_K - num_K_trunc)];
        num_ne_iter = 1;
        % Display status and store history of policies
        fprintf('Iteration %d policy: ', num_ne_iter);
        for i_u = 1 : ne_param.num_U
            for i_k = 1 : num_K
                if i_k == 1
                    fprintf('\n%d', m_down_u_k(i_u,i_k));
                else
                    fprintf('->%d', m_down_u_k(i_u,i_k));
                end
            end
        end
        fprintf('\n\n');
        m_down_u_k_hist = reshape(m_down_u_k.', 1, []);
        num_K_hist = num_K;
        limit_cycle = false;
        while ~ne_converged
            %% Get probability distribution over seeing messages
            M = unique(m_down_u_k);
            num_M = length(M);
            iota_up_m = zeros(num_M, 1);
            for i_u = 1 : ne_param.num_U
                if ne_param.U(i_u) == 0
                    iota_up_m(1) = iota_up_m(1) + p_up_u(i_u);
                else
                    for i_m = 1 : num_M
                        iota_up_m(i_m) = iota_up_m(i_m) + p_up_u(i_u) * sum(s_up_k(m_down_u_k(i_u,:) == M(i_m)));
                    end
                end
            end

            %% Get karma transition matrices for non-urgent and urgent agents
            % Matrices are 'fat'. We consider only next karma values that are possible
            kn_max = k_max + M(end);
            num_Kn = kn_max + 1;
            theta_down_u_k_up_kn = zeros(ne_param.num_U, num_K, num_Kn);
            for i_u = 1 : ne_param.num_U
                for i_k = 1 : num_K
                    m = m_down_u_k(i_u,i_k);
                    if m == 0
                        theta_down_u_k_up_kn(i_u,i_k,i_k+M) = iota_up_m.';
                    else
                        i_m_win = find(M < m);
                        i_kn_win = i_k - m;
                        theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = sum(iota_up_m(i_m_win));
                        i_m_tie = find(M == m);
                        if ~isempty(i_m_tie)
                            i_kn_tie = i_k + m;
                            theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = theta_down_u_k_up_kn(i_u,i_k,i_kn_win) + 0.5 * iota_up_m(i_m_tie);
                            theta_down_u_k_up_kn(i_u,i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
                        end
                        i_m_lose = find(M > m);
                        if ~isempty(i_m_lose)
                            i_kn_lose = i_k + M(i_m_lose);
                            theta_down_u_k_up_kn(i_u,i_k,i_kn_lose) = iota_up_m(i_m_lose).';
                        end
                    end
                end

            end
            t_down_k_up_kn = zeros(num_K, num_Kn);
            for i_u = 1 : ne_param.num_U
                t_down_k_up_kn = t_down_k_up_kn + p_up_u(i_u) * squeeze(theta_down_u_k_up_kn(i_u,:,:));
            end

            %% Now take a step in the direction of transition matrix
            s_up_k_next = t_down_k_up_kn.' * s_up_k;

            %% Now pluck out unlikely states form the end
            while s_up_k_next(end) < 1e-8
                s_up_k_next(end) = [];
            end

            %% Renormalize
            s_up_k_next = s_up_k_next / sum(s_up_k_next);

            %% Repeat all of the above until convegrence
            if length(s_up_k) == length(s_up_k_next)
                s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
            else
                s_converged = false;
            end
            s_up_k = s_up_k_next;
            while ~s_converged
                k_max = length(s_up_k) - 1;
                K = (0 : k_max).';
                num_K_prev = num_K;
                num_K = length(K);

                %% Get probability distribution over seeing messages
                if num_K > num_K_prev
                    m_down_u_k = [m_down_u_k, repmat(m_down_u_k(:,end), 1, num_K - num_K_prev)];
                elseif num_K < num_K_prev
                    m_down_u_k(:,num_K+1:end) = [];
                end
                M = unique(m_down_u_k);
                num_M = length(M);
                iota_up_m = zeros(num_M, 1);
                for i_u = 1 : ne_param.num_U
                    if ne_param.U(i_u) == 0
                        iota_up_m(1) = iota_up_m(1) + p_up_u(i_u);
                    else
                        for i_m = 1 : num_M
                            iota_up_m(i_m) = iota_up_m(i_m) + p_up_u(i_u) * sum(s_up_k(m_down_u_k(i_u,:) == M(i_m)));
                        end
                    end
                end

                %% Get karma transition matrices for non-urgent and urgent agents
                % Matrices are 'fat'. We consider only next karma values that are possible
                kn_max = k_max + M(end);
                num_Kn = kn_max + 1;
                theta_down_u_k_up_kn = zeros(ne_param.num_U, num_K, num_Kn);
                for i_u = 1 : ne_param.num_U
                    for i_k = 1 : num_K
                        m = m_down_u_k(i_u,i_k);
                        if m == 0
                            theta_down_u_k_up_kn(i_u,i_k,i_k+M) = iota_up_m.';
                        else
                            i_m_win = find(M < m);
                            i_kn_win = i_k - m;
                            theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = sum(iota_up_m(i_m_win));
                            i_m_tie = find(M == m);
                            if ~isempty(i_m_tie)
                                i_kn_tie = i_k + m;
                                theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = theta_down_u_k_up_kn(i_u,i_k,i_kn_win) + 0.5 * iota_up_m(i_m_tie);
                                theta_down_u_k_up_kn(i_u,i_k,i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
                            end
                            i_m_lose = find(M > m);
                            if ~isempty(i_m_lose)
                                i_kn_lose = i_k + M(i_m_lose);
                                theta_down_u_k_up_kn(i_u,i_k,i_kn_lose) = iota_up_m(i_m_lose).';
                            end
                        end
                    end

                end
                t_down_k_up_kn = zeros(num_K, num_Kn);
                for i_u = 1 : ne_param.num_U
                    t_down_k_up_kn = t_down_k_up_kn + p_up_u(i_u) * squeeze(theta_down_u_k_up_kn(i_u,:,:));
                end

                %% Now take a step in the direction of transition matrix
                s_up_k_next = t_down_k_up_kn.' * s_up_k;

                %% Now pluck out unlikely states form the end
                while s_up_k_next(end) < 1e-8
                    s_up_k_next(end) = [];
                end

                %% Renormalize
                s_up_k_next = s_up_k_next / sum(s_up_k_next);

                %% Repeat all of the above until convegrence
                if length(s_up_k) == length(s_up_k_next)
                    s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
                else
                    s_converged = false;
                end
                s_up_k = s_up_k_next;
            end

            %% Get expected stage cost
            q_down_u_k = zeros(ne_param.num_U, num_K);
            for i_u = 1 : ne_param.num_U
                if ne_param.U(i_u) == 0
                    continue;
                end
                for i_k = 1 : num_K
                    m = m_down_u_k(i_u,i_k);
                    i_m_tie = find(M == m);
                    if ~isempty(i_m_tie)
                        q_down_u_k(i_u,i_k) = 0.5 * iota_up_m(i_m_tie) * ne_param.U(i_u);
                    end
                    i_m_lose = find(M > m);
                    q_down_u_k(i_u,i_k) = q_down_u_k(i_u,i_k) + sum(iota_up_m(i_m_lose)) * ne_param.U(i_u);
                end
            end

            %% Concatenate
            % Get rid of 'extra' states. Is this the same as saturation?
            theta_down_u_k_up_kn_trunc = theta_down_u_k_up_kn(:,:,1:num_K);
            theta_down_u_k_up_kn_trunc(:,:,end) = 1 - sum(theta_down_u_k_up_kn_trunc(:,:,1:end-1), 3);
            t_down_u_k_up_un_kn = zeros(ne_param.num_U, num_K, ne_param.num_U, num_K);
            for i_u = 1 : ne_param.num_U
                for i_un = 1 : ne_param.num_U
                    t_down_u_k_up_un_kn(i_u,:,i_un,:) = ne_param.mu_down_u_up_un(i_u,i_un) * squeeze(theta_down_u_k_up_kn_trunc(i_u,:,:));
                end
            end

            %% Get expected infinite horizon cost
            v_down_u_k = zeros(ne_param.num_U, num_K);
            v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
            num_v_iter = 1;
            v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
            v_down_u_k = v_down_u_k_next;
            while ~v_converged
                v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
                num_v_iter = num_v_iter + 1;
                v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
                v_down_u_k = v_down_u_k_next;
            end

            %% Get best response
            % Only do so for 'prevalent' k's
            s_up_k_trunc = s_up_k;
            while s_up_k_trunc(end) < 1e-3
                s_up_k_trunc(end) = [];
            end
            num_K_trunc = length(s_up_k_trunc);
            br_m_down_u_k = m_down_u_k(:,1:num_K_trunc);
            for i_u = 1 : ne_param.num_U
                if ne_param.U(i_u) == 0
                    continue;
                end
                for i_k = 1 : num_K_trunc
                    if K(i_k) == 0
                        continue;
                    end
                    rho_down_u_k_m_min = v_down_u_k(i_u,i_k);
                    for m = br_m_down_u_k(i_u,i_k-1) : K(i_k)
                        if m == m_down_u_k(i_u,i_k)
                            continue;
                        end
                        % Expected stage cost for current message
                        i_m_tie = find(M == m);
                        if isempty(i_m_tie)
                            eta_down_u_k_m = 0;
                        else
                            eta_down_u_k_m = 0.5 * iota_up_m(i_m_tie) * ne_param.U(i_u);
                        end
                        i_m_lose = find(M > m);
                        eta_down_u_k_m = eta_down_u_k_m + sum(iota_up_m(i_m_lose)) * ne_param.U(i_u);
                        % Expected transition probabilities for current message
                        tau_down_u_k_m_up_kn = zeros(1, num_Kn);
                        if m == 0
                            tau_down_u_k_m_up_kn(i_k+M) = iota_up_m.';
                        else
                            i_m_win = find(M < m);
                            i_kn_win = i_k - m;
                            tau_down_u_k_m_up_kn(i_kn_win) = sum(iota_up_m(i_m_win));
                            if ~isempty(i_m_tie)
                                i_kn_tie = i_k + m;
                                tau_down_u_k_m_up_kn(i_kn_win) = tau_down_u_k_m_up_kn(i_kn_win) + 0.5 * iota_up_m(i_m_tie);
                                tau_down_u_k_m_up_kn(i_kn_tie) = 0.5 * iota_up_m(i_m_tie);
                            end
                            if ~isempty(i_m_lose)
                                i_kn_lose = i_k + M(i_m_lose);
                                tau_down_u_k_m_up_kn(i_kn_lose) = iota_up_m(i_m_lose).';
                            end
                        end

                        tau_down_u_k_m_up_kn_trunc = tau_down_u_k_m_up_kn(1:num_K);
                        tau_down_u_k_m_up_kn_trunc(end) = 1 - sum(tau_down_u_k_m_up_kn_trunc(1:end-1));
                        chi_down_u_k_m_up_un_kn = zeros(ne_param.num_U, num_K);
                        for i_un = 1 : ne_param.num_U
                            chi_down_u_k_m_up_un_kn(i_un,:) = ne_param.mu_down_u_up_un(i_u,i_un) * tau_down_u_k_m_up_kn_trunc;
                        end
                        rho_down_u_k_m = eta_down_u_k_m + alpha * reshape(chi_down_u_k_m_up_un_kn, [], 1).' * reshape(v_down_u_k, [], 1);
                        if rho_down_u_k_m < rho_down_u_k_m_min - ne_param.br_v_tol
                            br_m_down_u_k(i_u,i_k) = m;
                            rho_down_u_k_m_min = rho_down_u_k_m;
                        end
                    end
                    if br_m_down_u_k(i_u,i_k) < br_m_down_u_k(i_u,i_k-1)
                        br_m_down_u_k(i_u,i_k) = br_m_down_u_k(i_u,i_k-1);
                    end
                end
            end

            ne_converged = norm(m_down_u_k(:,1:num_K_trunc) - br_m_down_u_k, inf) <= ne_param.br_pi_tol;
            m_down_u_k = [br_m_down_u_k, repmat(br_m_down_u_k(:,end), 1, num_K - num_K_trunc)];
            num_ne_iter = num_ne_iter + 1;
            % Display status and store history of policies
            fprintf('Iteration %d policy: ', num_ne_iter);
            for i_u = 1 : ne_param.num_U
                for i_k = 1 : num_K
                    if i_k == 1
                        fprintf('\n%d', m_down_u_k(i_u,i_k));
                    else
                        fprintf('->%d', m_down_u_k(i_u,i_k));
                    end
                end
            end
            fprintf('\n\n');

            % Process history
            if num_K_hist < num_K
                m_down_u_k_hist_old = m_down_u_k_hist;
                m_down_u_k_hist = [];
                for i_u = 1 : ne_param.U
                    i_u_base = (i_u - 1) * num_K_hist;
                    m_down_u_k_hist = [m_down_u_k_hist, m_down_u_k_hist_old(:,i_u_base+1:i_u_base+num_K_hist), repmat(m_down_u_k_hist_old(:,i_u_base+num_K_hist), 1, num_K - num_K_hist)];
                end
                m_down_u_k_hist_end = reshape(m_down_u_k.', 1, []);
                num_K_hist = num_K;
            elseif num_K_hist > num_K
                m_down_u_k_hist_end = [m_down_u_k, repmat(m_down_u_k(:,end), 1, num_K_hist - num_K)];
                m_down_u_k_hist_end = reshape(m_down_u_k_hist_end.', 1, []);
            else
                m_down_u_k_hist_end = reshape(m_down_u_k.', 1, []);
            end

            % Detect a limit cycle
            for hist_i = 1 : size(m_down_u_k_hist, 1)
                if isequal(m_down_u_k_hist(hist_i,:), m_down_u_k_hist_end)
                    % Limit cycle found
                    m_down_u_k_limit_cycle = m_down_u_k_hist(hist_i:end,:);
                    if size(m_down_u_k_limit_cycle, 1) > 1
                        limit_cycle = true;
                    end
                    break;
                end
            end
            m_down_u_k_hist = [m_down_u_k_hist; m_down_u_k_hist_end];
            if  limit_cycle
                fprintf('Limit cycle found!\n\n');
                break;
            end
        end

        %% Result
        if  limit_cycle
            % Mix limit cycle policies uniformly
            num_limit_cycle = size(m_down_u_k_limit_cycle, 1);
            ne_pi_down_u_k_up_m = zeros(ne_param.num_U, num_K_hist, num_K_hist);
            for i_u = 1 : ne_param.num_U
                i_u_base = (i_u - 1) * num_K_hist;
                for i_k = 1 : num_K_hist
                    for i_limit_cycle = 1 : num_limit_cycle
                        i_m = find(K == m_down_u_k_limit_cycle(i_limit_cycle,i_u_base+i_k));
                        ne_pi_down_u_k_up_m(i_u,i_k,i_m) = ne_pi_down_u_k_up_m(i_u,i_k,i_m) + 1 / num_limit_cycle;
                    end
                end
            end

            %% Get probability distribution over seeing messages
            d_up_u_k = outer(p_up_u, s_up_k);
            iota_up_m = squeeze(dot2(reshape(d_up_u_k, [], 1), reshape(ne_pi_down_u_k_up_m(:,1:num_K,1:num_K), [], num_K), 1, 1)).';
            while(iota_up_m(end) < 1e-8)
                iota_up_m(end) = [];
            end
            iota_up_m = iota_up_m / sum(iota_up_m);
            num_M = length(iota_up_m);
            M = (0 : num_M - 1).';

            %% Get karma transition matrices for non-urgent and urgent agents
            % Matrices are 'fat'. We consider only next karma values that are possible
            kn_max = k_max + M(end);
            num_Kn = kn_max + 1;
            theta_down_u_k_up_kn = zeros(ne_param.num_U, num_K, num_Kn);
            for i_u = 1 : ne_param.num_U
                for i_k = 1 : num_K
                    for i_m = 1 : num_M
                        p = ne_pi_down_u_k_up_m(i_u,i_k,i_m); 
                        if p == 0
                            continue;
                        end
                        m = M(i_m);
                        if m == 0
                            theta_down_u_k_up_kn(i_u,i_k,i_k+M) = squeeze(theta_down_u_k_up_kn(i_u,i_k,i_k+M)).' + p * iota_up_m.';
                        else
                            i_m_win = find(M < m);
                            i_kn_win = i_k - m;
                            theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = theta_down_u_k_up_kn(i_u,i_k,i_kn_win) + p * sum(iota_up_m(i_m_win));
                            i_m_tie = find(M == m);
                            if ~isempty(i_m_tie)
                                i_kn_tie = i_k + m;
                                theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = theta_down_u_k_up_kn(i_u,i_k,i_kn_win) + p * 0.5 * iota_up_m(i_m_tie);
                                theta_down_u_k_up_kn(i_u,i_k,i_kn_tie) = theta_down_u_k_up_kn(i_u,i_k,i_kn_tie) + p * 0.5 * iota_up_m(i_m_tie);
                            end
                            i_m_lose = find(M > m);
                            if ~isempty(i_m_lose)
                                i_kn_lose = i_k + M(i_m_lose);
                                theta_down_u_k_up_kn(i_u,i_k,i_kn_lose) = squeeze(theta_down_u_k_up_kn(i_u,i_k,i_kn_lose)).' + p * iota_up_m(i_m_lose).';
                            end
                        end
                    end
                end
            end
            t_down_k_up_kn = zeros(num_K, num_Kn);
            for i_u = 1 : ne_param.num_U
                t_down_k_up_kn = t_down_k_up_kn + p_up_u(i_u) * squeeze(theta_down_u_k_up_kn(i_u,:,:));
            end

            %% Now take a step in the direction of transition matrix
            s_up_k_next = t_down_k_up_kn.' * s_up_k;

            %% Now pluck out unlikely states form the end
            while s_up_k_next(end) < 1e-8
                s_up_k_next(end) = [];
            end

            %% Renormalize
            s_up_k_next = s_up_k_next / sum(s_up_k_next);

            %% Repeat all of the above until convegrence
            if length(s_up_k) == length(s_up_k_next)
                s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
            else
                s_converged = false;
            end
            s_up_k = s_up_k_next;
            while ~s_converged
                k_max = length(s_up_k) - 1;
                K = (0 : k_max).';
                num_K_prev = num_K;
                num_K = length(K);

                %% Get probability distribution over seeing messages
                if num_K > num_K_hist
                    ne_pi_down_u_k_up_m_old = ne_pi_down_u_k_up_m;
                    ne_pi_down_u_k_up_m = zeros(ne_param.num_U, num_K, num_K);
                    ne_pi_down_u_k_up_m(:,1:num_K_hist,1:num_K_hist) = ne_pi_down_u_k_up_m_old;
                    ne_pi_down_u_k_up_m(:,num_K_hist+1:end,:) = ne_pi_down_u_k_up_m(:,num_K_hist,:);
                    num_K_hist = num_K;
                end
                d_up_u_k = outer(p_up_u, s_up_k);
                iota_up_m = squeeze(dot2(reshape(d_up_u_k, [], 1), reshape(ne_pi_down_u_k_up_m(:,1:num_K,1:num_K), [], num_K), 1, 1)).';
                while(iota_up_m(end) < 1e-8)
                    iota_up_m(end) = [];
                end
                iota_up_m = iota_up_m / sum(iota_up_m);
                num_M = length(iota_up_m);
                M = (0 : num_M - 1).';

                %% Get karma transition matrices for non-urgent and urgent agents
                % Matrices are 'fat'. We consider only next karma values that are possible
                kn_max = k_max + M(end);
                num_Kn = kn_max + 1;
                theta_down_u_k_up_kn = zeros(ne_param.num_U, num_K, num_Kn);
                for i_u = 1 : ne_param.num_U
                    for i_k = 1 : num_K
                        for i_m = 1 : num_M
                            p = ne_pi_down_u_k_up_m(i_u,i_k,i_m); 
                            if p == 0
                                continue;
                            end
                            m = M(i_m);
                            if m == 0
                                theta_down_u_k_up_kn(i_u,i_k,i_k+M) = squeeze(theta_down_u_k_up_kn(i_u,i_k,i_k+M)).' + p * iota_up_m.';
                            else
                                i_m_win = find(M < m);
                                i_kn_win = i_k - m;
                                theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = theta_down_u_k_up_kn(i_u,i_k,i_kn_win) + p * sum(iota_up_m(i_m_win));
                                i_m_tie = find(M == m);
                                if ~isempty(i_m_tie)
                                    i_kn_tie = i_k + m;
                                    theta_down_u_k_up_kn(i_u,i_k,i_kn_win) = theta_down_u_k_up_kn(i_u,i_k,i_kn_win) + p * 0.5 * iota_up_m(i_m_tie);
                                    theta_down_u_k_up_kn(i_u,i_k,i_kn_tie) = theta_down_u_k_up_kn(i_u,i_k,i_kn_tie) + p * 0.5 * iota_up_m(i_m_tie);
                                end
                                i_m_lose = find(M > m);
                                if ~isempty(i_m_lose)
                                    i_kn_lose = i_k + M(i_m_lose);
                                    theta_down_u_k_up_kn(i_u,i_k,i_kn_lose) = squeeze(theta_down_u_k_up_kn(i_u,i_k,i_kn_lose)).' + p * iota_up_m(i_m_lose).';
                                end
                            end
                        end
                    end
                end
                t_down_k_up_kn = zeros(num_K, num_Kn);
                for i_u = 1 : ne_param.num_U
                    t_down_k_up_kn = t_down_k_up_kn + p_up_u(i_u) * squeeze(theta_down_u_k_up_kn(i_u,:,:));
                end

                %% Now take a step in the direction of transition matrix
                s_up_k_next = t_down_k_up_kn.' * s_up_k;

                %% Now pluck out unlikely states form the end
                while s_up_k_next(end) < 1e-8
                    s_up_k_next(end) = [];
                end

                %% Renormalize
                s_up_k_next = s_up_k_next / sum(s_up_k_next);

                %% Repeat all of the above until convegrence
                if length(s_up_k) == length(s_up_k_next)
                    s_converged = norm(s_up_k_next - s_up_k, inf) <= ne_param.d_tol;
                else
                    s_converged = false;
                end
                s_up_k = s_up_k_next;
            end

            %% Get expected stage cost
            q_down_u_k = zeros(ne_param.num_U, num_K);
            for i_u = 1 : ne_param.num_U
                if ne_param.U(i_u) == 0
                    continue;
                end
                for i_k = 1 : num_K
                    for i_m = 1 : num_M
                        p = ne_pi_down_u_k_up_m(i_u,i_k,i_m); 
                        if p == 0
                            continue;
                        end
                        m = M(i_m);
                        i_m_tie = find(M == m);
                        if ~isempty(i_m_tie)
                            q_down_u_k(i_u,i_k) = q_down_u_k(i_u,i_k) + p * 0.5 * iota_up_m(i_m_tie) * ne_param.U(i_u);
                        end
                        i_m_lose = find(M > m);
                        q_down_u_k(i_u,i_k) = q_down_u_k(i_u,i_k) + p * sum(iota_up_m(i_m_lose)) * ne_param.U(i_u);
                    end
                end
            end

            %% Concatenate
            % Get rid of 'extra' states. Is this the same as saturation?
            theta_down_u_k_up_kn_trunc = theta_down_u_k_up_kn(:,:,1:num_K);
            theta_down_u_k_up_kn_trunc(:,:,end) = 1 - sum(theta_down_u_k_up_kn_trunc(:,:,1:end-1), 3);
            t_down_u_k_up_un_kn = zeros(ne_param.num_U, num_K, ne_param.num_U, num_K);
            for i_u = 1 : ne_param.num_U
                for i_un = 1 : ne_param.num_U
                    t_down_u_k_up_un_kn(i_u,:,i_un,:) = ne_param.mu_down_u_up_un(i_u,i_un) * squeeze(theta_down_u_k_up_kn_trunc(i_u,:,:));
                end
            end

            %% Get expected infinite horizon cost
            v_down_u_k = zeros(ne_param.num_U, num_K);
            v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
            num_v_iter = 1;
            v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
            v_down_u_k = v_down_u_k_next;
            while ~v_converged
                v_down_u_k_next = q_down_u_k + alpha * dot2(reshape(t_down_u_k_up_un_kn, ne_param.num_U, num_K, []), reshape(v_down_u_k, [], 1), 3, 1);
                num_v_iter = num_v_iter + 1;
                v_converged = norm(v_down_u_k_next - v_down_u_k, inf) <= ne_param.v_tol;
                v_down_u_k = v_down_u_k_next;
            end

            ne_pi_down_u_k_up_m = ne_pi_down_u_k_up_m(:,1:num_K,1:num_K);
        else
            % We have a pure NE policy
            ne_pi_down_u_k_up_m = zeros(ne_param.num_U, num_K, num_K);
            for i_u = 1 : ne_param.num_U
                for i_k = 1 : num_K
                    i_m = find(K == m_down_u_k(i_u,i_k));
                    ne_pi_down_u_k_up_m(i_u,i_k,i_m) = 1;
                end
            end
        end
        ne_K = K;
        ne_p_up_u = p_up_u;
        ne_s_up_k = s_up_k;
        ne_d_up_u_k = outer(p_up_u, s_up_k);
        ne_q_down_u_k = q_down_u_k;
        ne_v_down_u_k = v_down_u_k;
        ne_t_down_u_k_up_un_kn = t_down_u_k_up_un_kn;
        
        %% Store end results
        if ne_param.save
            save(['karma_nash_equilibrium/results/k_ave_', num2str(k_ave, '%02d'), '_alpha_', num2str(alpha, '%.2f'), '.mat']);
        end
    end
end

%% Plot
% NE policy plot
ne_pi_plot_fg = 1;
ne_pi_plot_pos = [0, default_height, default_width, default_height];
ne_func.plot_ne_pi(ne_pi_plot_fg, ne_pi_plot_pos, RedColormap, ne_pi_down_u_k_up_m, ne_param.U, ne_K, ne_K, k_ave, alpha);

% NE stationary distribution plot
ne_d_plot_fg = 2;
ne_d_plot_pos = [0, 0, default_width, default_height];
ne_func.plot_ne_d(ne_d_plot_fg, ne_d_plot_pos, ne_d_up_u_k, ne_param.U, ne_K, k_ave, alpha);

% NE expected utility plot
ne_v_plot_fg = 3;
ne_v_plot_pos = [default_width, default_height, default_width, default_height];
ne_func.plot_ne_v(ne_v_plot_fg, ne_v_plot_pos, ne_v_down_u_k, ne_param.U, ne_K, k_ave, alpha);

% NE state transitions plot
ne_t_plot_fg = 4;
ne_t_plot_pos = [0, 0, screenwidth, screenheight];
ne_func.plot_ne_t(ne_t_plot_fg, ne_t_plot_pos, RedColormap, ne_t_down_u_k_up_un_kn, ne_param.U, ne_K, k_ave, alpha);

%% Inform user when done
fprintf('DONE\n\n');