function [idx_best_arm, opt_policies] = policy_ucb_tuned(arms, p0, p1, t, opt_policies)
%POLICY_UCB_TUNED select the best arm for the t turn according to UCB1-Tuned policy
%
%   Input:
%       arms: arms of the bandit problem
%       p0: vector containing the number of fails of each arm
%       p1: vector containing the number of successes of each arm
%       t: current iteration
%
%   Output:
%       idx_best_arm: the selected arm for the t turn of iteration
%

%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

n = p0 + p1;
m = p1./n;
V = (p0 .* (0 - m).^2 + p1 .* (1 - m).^2) ./ n + sqrt(2*log(t)./n);
v = arms .* (m + sqrt(log(t)./n .* min(1/4,V)));
[~,idx_best_arm] = max(v);
