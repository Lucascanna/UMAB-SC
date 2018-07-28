function [idx_best_arm, opt_policies] = policy_ucb_lm(arms, p0, p1, t, opt_policies, varargin)
%POLICY_UCB-PM select the best arm for the t turn according to UCB prior
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

n_arms = length(arms);

n = p0 + p1;
p_max = opt_policies.mu_max;

for kk = n_arms:-1:1
    % Search of optimal value for j, from kk to 1
    ord_n = n(kk:-1:1);
    ord_p1 = p1(kk:-1:1);
    temp_upper = cumsum(ord_p1) ./ cumsum(ord_n) + sqrt( 2 * p_max * ( 4 * log(t-1) + log(kk)) ./ cumsum(ord_n));
    upper(kk) = min([temp_upper, 1]);
end

% Value computation
v = upper .* arms;
[~,idx_best_arm] = max(v);

