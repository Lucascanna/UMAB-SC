function [idx_best_arm, opt_policies] = policy_dumb(arms, ~, ~, ~, opt_policies)
%POLICY_DUMB select the best arm for the t turn randomly
%
%   Input:
%       arms: arms of the bandit problem
%
%   Output:
%       idx_best_arm: the selected arm for the t turn of iteration
%

%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

idx_best_arm = randi(length(arms));
