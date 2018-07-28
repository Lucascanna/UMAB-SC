function [idx_best_arm, opt_policies] = policy_ucb2(arms,p0,p1,t,opt_policies)
%POLICY_UCB2 select the best arm for the t turn according to UCB2 policy
%
%   Input:
%       arms: arms of the bandit problem
%       p0: vector containing the number of fails of each arm
%       p1: vector containing the number of successes of each arm
%       t: current iteration
%       opt_policies.ucb2.epochs: number of epochs played by each arm
%
%   Output:
%       idx_best_arm: the selected arm for the t turn of iteration
%

%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

epochs = opt_policies.ucb2.epochs;
alpha = opt_policies.ucb2.alpha;

n = p0 + p1;
mu = p1./n;
tau_r = ceil((1+alpha).^epochs);
a = sqrt((1+alpha)*log(exp(1)*(t-1)./tau_r)./(2*tau_r));
v = arms .* (mu + a);
[~,idx_best_arm] = max(v);

opt_policies.ucb2.n_turns = ceil( (1+alpha)^(epochs(idx_best_arm)+1) ) - ceil( (1+alpha)^(epochs(idx_best_arm)) );
opt_policies.ucb2.epochs(idx_best_arm) = opt_policies.ucb2.epochs(idx_best_arm) + 1; % Update number of epoch of best_arm
