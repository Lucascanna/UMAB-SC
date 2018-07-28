function [idx_best_arm, opt_policies] = policy_bayes_ucb(arms, p0, p1, t, opt_policies)
%POLICY_BAYES_UCB select the best arm for the t turn according to Bayes-UCB policy
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

upper = betaincinv(1-1/t,p1+1,p0+1);
[~,idx_best_arm] = max(upper .* arms);
