clear;
close all;
clc;

%% Start with a NE guess policy function
% Functions take the following form:
% pi_down_0(k) = 0 (bid 0 if non-urgent)
% pi_down_U(k) =    round(a * k) if k <= b
%                   round(a * b + a * log(k - b + 1)) if k > b
% (linear then logarithmic)
% Note that policy functions are pure. They are characterized by parameters
% a & b
a = 0.5;
b = 2;
c = 3;

K = 0 : 0.1 : 42;

M_k_lin = a * K(K <= b);
M_k_log = a * b + c * log(K(K > b) - b + 1);
M_k = [M_k_lin, M_k_log];

plot(K, K, 'LineWidth', 2);
hold on;
plot(K, M_k);