function [idx_best_arm, opt_policies] = policy_thompson(arms, p0, p1, ~, opt_policies)
%POLICY_THOMPSON select the best arm for the t turn according to Thompson policy
%
%   Input:
%       arms: arms of the bandit problem
%       p0: vector containing the number of fails of each arm
%       p1: vector containing the number of successes of each arm
%
%   Output:
%       idx_best_arm: the selected arm for the t turn of iteration
%

%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milanoù

samples = betarnd(p1+1,p0+1);
v = samples .* arms;
[~,idx_best_arm] = max(v);
