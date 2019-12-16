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

norm(x1 + x2)