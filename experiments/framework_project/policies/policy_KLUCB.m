function [idx_best_arm, opt_policies] = policy_KLUCB(arms, p0, p1, t, opt_policies)
%POLICY_KLUCB select the best arm for the t turn according to KL-UCB policy
%
%   Input:
%       arms: arms of the bandit problem
%       p0: vector containing the number of fails of each arm
%       p1: vector containing the number of successes of each arm
%       t: current iteration
%       opt_policies.kl_ucb.c: the c parameter
%
%   Output:
%       idx_best_arm: the selected arm for the t turn of iteration
%

%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

n = p0 + p1;
n_arms = max(size(arms));

% KL maximization
limit = opt_policies.kl_ucb.c * log(log(t)) + log(t);
p = p1 ./ n;
q = zeros(size(arms));
for jj = 1:n_arms
    [~, q(jj)] = max_kldiv(p(jj), limit / n(jj), opt_policies.kl_ucb.max_tol);
end
v = q .* arms;
[~,idx_best_arm] = max(v);
