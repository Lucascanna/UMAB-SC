function [idx_best_arm, opt_policies] = policy_ucb_l(arms, p0, p1, t, opt_policies, varargin)
%POLICY_UCB1 select the best arm for the t turn according to UCB prior
%policy
%
%   Input:
%       arms: arms of the bandit problem
%       p0: vector containing the number of fails of each arm
%       p1: vector containing the number of successes of each arm
%       t: current iteration
%       opt_policies.ucb_prior.a: optimized ucb parameter
%       opt_policies.ucb_prior.b: optimized ucb parameter
%
%   Output:
%       idx_best_arm: the selected arm for the t turn of iteration
%
%   Copyright ...

mu_max = opt_policies.mu_max;
n = p0 + p1;
upper = p1 ./ n + sqrt(8 * mu_max * log(t-1) ./ n);
v = upper .* arms;
[~,idx_best_arm] = max(v);

