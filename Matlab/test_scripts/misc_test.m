clear;
clc;
close all;

n = 5;
x1 = rand(n, 1);
x1 = x1 / sum(x1);
% x1 = zeros(n, 1);
% x1(1) = 1;
x2 = rand(n, 1);
x2 = x2 / sum(x2);
% x2 = zeros(n, 1);
% x2(1) = 0.9;
% x2(2) = 0.1;

x1_x1T = x1 * x1.';
x1_x2T = x1 * x2.';

eig_x1_x1T = eig(x1_x1T);
eig_x1_x2T = eig(x1_x2T);

norm(x1 + x2);

M = 0 : 0.01 : 12;
num_M = length(M);
mj = 6;
delta_0_1 = zeros(1, num_M);
delta_norm = zeros(1, num_M);
delta_s_norm = 0.75;
delta_log = zeros(1, num_M);
delta_s_log = 0.5;
for i_m = 1 : num_M
    m = M(i_m);
    delta_0_1(i_m) = max([0, min([(m - mj + 1) / 2, 1])]);
    delta_norm(i_m) = 0.5 * (1 + erf((m - mj) / (sqrt(2) * delta_s_norm)));
    delta_log(i_m) = 1 / (1 + exp(-(m - mj) / delta_s_log));
end
i_plot = find(M >= 3 & M <= 9);
figure(1);
plot(M(i_plot), delta_0_1(i_plot));
hold on;
plot(M(i_plot), delta_norm(i_plot));
plot(M(i_plot), delta_log(i_plot));
stem(mj, 1, 'Marker', 'none');
axis tight;
axes = gca;
axes.Title.FontName = 'ubuntu';
axes.Title.String = 'Probability of winning bid';
axes.Title.FontSize = 12;
axes.XAxis.FontSize = 10;
axes.YAxis.FontSize = 10;
axes.XLabel.FontName = 'ubuntu';
axes.XLabel.String = 'Message';
axes.XLabel.FontSize = 12;
axes.YLabel.FontName = 'ubuntu';
axes.YLabel.String = '\delta(m,m^j)';
axes.YLabel.FontSize = 12;
lgd = legend('\delta_{0,1}', '\delta_{norm}', '\delta_{log}', 'm^j = 6');
lgd.FontSize = 12;
lgd.FontName = 'ubuntu';
lgd.Location = 'bestoutside';

Mb = 0 : 12;
num_Mb = length(Mb);
beta_0_1 = zeros(num_Mb, num_M);
beta_norm = zeros(num_Mb, num_M);
beta_s_norm = 0.5;
beta_log = zeros(num_Mb, num_M);
beta_s_log = 0.3;
for i_m = 1 : num_M
    m = M(i_m);
    for i_mb = 1 : num_Mb
        mb = Mb(i_mb);
        beta_0_1(i_mb,i_m) = max([0, 1 - abs(mb - m)]);
        beta_norm(i_mb,i_m) = exp(-(mb - m)^2 / (2 * beta_s_norm^2));
        beta_log(i_mb,i_m) = exp(-(mb - m) / beta_s_log) / ((1 + exp(-(mb - m) / beta_s_log))^2);
    end
    beta_norm(:,i_m) = beta_norm(:,i_m) / sum(beta_norm(:,i_m));
    beta_log(:,i_m) = beta_log(:,i_m) / sum(beta_log(:,i_m));
end
figure(2);
i_mb = find(Mb == 6);
plot(M(i_plot), beta_0_1(i_mb,i_plot));
hold on;
plot(M(i_plot), beta_norm(i_mb,i_plot));
plot(M(i_plot), beta_log(i_mb,i_plot));
stem(mj, 1, 'Marker', 'none');
axis tight;
axes = gca;
axes.Title.FontName = 'ubuntu';
axes.Title.String = 'Probability of exchanging karma';
axes.Title.FontSize = 12;
axes.XAxis.FontSize = 10;
axes.YAxis.FontSize = 10;
axes.XLabel.FontName = 'ubuntu';
axes.XLabel.String = 'Message';
axes.XLabel.FontSize = 12;
axes.YLabel.Interpreter = 'latex';
axes.YLabel.String = '$\beta(\bar{m},m)$';
axes.YLabel.FontSize = 12;
lgd = legend('$\beta_{0,1}$', '$\beta_{norm}$', '$\beta_{log}$', '$\bar{m} = 6$');
lgd.FontSize = 12;
lgd.Interpreter = 'latex';
lgd.Location = 'bestoutside';