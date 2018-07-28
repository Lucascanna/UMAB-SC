function [idx_best_arm,opt_policies] = policy_ucb1_m(arms,p0,p1,t,opt_policies,varargin)
%POLICY_UCB_ORDERED select the best arm for the t turn according to UCB-Ordered policy
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
%   Copyright ...

n_arms = length(arms);

n = p0 + p1;

for kk = n_arms:-1:1
    % Search of optimal value for j, from kk to 1
    ord_n = n(kk:-1:1);
    ord_p1 = p1(kk:-1:1);
    temp_upper = cumsum(ord_p1) ./ cumsum(ord_n) + sqrt( (4 * log(t-1) + log(kk)) / 2 ./ cumsum(ord_n) );
    upper(kk) = min([temp_upper, 1]);
end

% Value computation
v = upper .* arms;

[~,idx_best_arm] = max(v);