function [idx_best_arm, opt_policies] = policy_ucb1(arms, p0, p1, t, opt_policies)
%POLICY_UCB1 select the best arm for the t turn according to UCB1 policy
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
v = (p1./n + sqrt(2*log(t)./n)) .* arms;
[~,idx_best_arm] = max(v);
