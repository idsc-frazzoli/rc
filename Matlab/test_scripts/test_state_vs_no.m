clc;
clear all;
close all;

alpha = 0.8;
p_u_l = rand(1);
p_u_h = 1 - p_u_l;
% p_i_j_k = p(k(t+1)=i|u(t)=j,k(t)=k) in the following
p_0_0_0 = rand(1);
p_1_0_0 = 1 - p_0_0_0;
p_0_0_1 = rand(1);
p_1_0_1 = 1 - p_0_0_1;
p_0_1_0 = rand(1);
p_1_1_0 = 1 - p_0_1_0;
p_0_1_1 = rand(1);
p_1_1_1 = 1 - p_0_1_1;

T = [p_u_l * p_0_0_0, p_u_l * p_1_0_0, p_u_h * p_0_0_0, p_u_h * p_1_0_0;...
    p_u_l * p_0_0_1, p_u_l * p_1_0_1, p_u_h * p_0_0_1, p_u_h * p_1_0_1;...
    p_u_l * p_0_1_0, p_u_l * p_1_1_0, p_u_h * p_0_1_0, p_u_h * p_1_1_0;...
    p_u_l * p_0_1_1, p_u_l * p_1_1_1, p_u_h * p_0_1_1, p_u_h * p_1_1_1];

c = rand(4,1) * 3;

theta = (eye(4) - alpha * T) \ c;

T_u_l = T(1:2,1:2) + T(1:2,2+1:end);
T_u_h = T(2+1:end,1:2) + T(2+1:end,2+1:end);
T_k = p_u_l * T_u_l + p_u_h * T_u_h;

c_k = [p_u_l * c(1) + p_u_h * c(3); p_u_l * c(2) + p_u_h * c(4)];

theta_k = (eye(2) - alpha * T_k) \ c_k

theta_k_orig = [p_u_l * theta(1) + p_u_h * theta(3); p_u_l * theta(2) + p_u_h * theta(4)]