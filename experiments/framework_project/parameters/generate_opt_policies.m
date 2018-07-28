function opt_policies = generate_opt_policies()
%GENERATE_OPT_POLICIES generate the options for the policies which needs some parameters
%

%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

% UCB2
opt_policies.ucb2.epochs = []; % epochs are initialized in run_configuration
opt_policies.ucb2.n_turns = 0;
opt_policies.ucb2.alpha = 1e-1; % parameter between 0 and 1. (1/8 * (7-sqrt(33))

% KL UCB
opt_policies.kl_ucb.c = 0;
opt_policies.kl_ucb.max_tol = 1e-10; %tolerance for the kl divergence maximization
