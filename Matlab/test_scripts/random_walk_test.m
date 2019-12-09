clc;
clear all;
close all;

num_agents = 200;
num_iter = 10000;
x = zeros(num_iter, num_agents);

for i = 2 : num_iter
    x(i,:) = x(i-1,:) + (rand(1, num_agents) > 0.5) - 0.5;
end
variance = var(x, [], 2);
variance_div_time = variance ./ (1 : num_iter).';

figure;
plot(x);
figure;
plot(variance);
figure
plot(variance_div_time);
