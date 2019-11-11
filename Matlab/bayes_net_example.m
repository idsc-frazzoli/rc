%% Implements example 'Cloudy-Sprinkler-Rain-WetGrass' Bayes net
clear;
close all;
clc;

%% Graph structure
N = 4;
dag = false(N);
C = 1;
S = 2;
R = 3;
W = 4;
dag(C,[R S]) = true;
dag(R,W) = true;
dag(S,W) = true;

%% Define type and size of nodes
discrete_nodes = 1 : N;
node_sizes = 2 * ones(1, N);

%% Make Bayes net
bnet = mk_bnet(dag, node_sizes);

%% Conditional probability distributions
bnet.CPD{C} = tabular_CPD(bnet, C, [0.5 0.5]);
bnet.CPD{S} = tabular_CPD(bnet, S, [0.5 0.9 0.5 0.1]);
bnet.CPD{R} = tabular_CPD(bnet, R, [0.8 0.2 0.2 0.8]);
bnet.CPD{W} = tabular_CPD(bnet, W, [1 0.1 0.1 0.01 0 0.9 0.9 0.99]);

%% Visualize network
G = bnet.dag;
draw_graph(G);

%% Inference
engine = jtree_inf_engine(bnet);

% Compute marginal distributions with no evidence
evidence = cell(1,N);
[engine, loglik] = enter_evidence(engine, evidence);
marg_C = marginal_nodes(engine, C);
marg_S = marginal_nodes(engine, S);
marg_R = marginal_nodes(engine, R);
marg_W = marginal_nodes(engine, W);