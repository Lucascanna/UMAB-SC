% Script to execute experiments on different pricing configurations by
% considering different MAB policies
%
%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

clear
clc
close all
addpath(genpath('.'));

%% SETTING
% Experimental setting
n_iterations = 10000;
n_fix = 1;          % number of steps you want to keep the same arm
n_repetitions = 2;  % must be >= 2, in order to evaluate the standard deviation
conf_idx = 1:3;    % indexes of the configurations you want to test
n_configurations = numel(conf_idx);
mu_max = 10^(-1);

% Choice of the policies
[policies, names] = set_policies(); % the policies you want to test
n_policies = numel(policies);

% Name of the data file
file_signature = [num2str(n_configurations) '_' num2str(n_repetitions) '_' num2str(n_iterations) '_' date];

% Selection of the configurations
% configuration = generate_configurations(); % Configuration without mu_max
configuration = generate_configurations_mu_max(mu_max); % Configuration with mu_max
configuration = configuration(conf_idx);

% Generation of the thresholds
[thresholds, values] = generate_thresholds(configuration, n_configurations, n_repetitions, n_iterations);
mkdir(['.' filesep 'res_' file_signature filesep]);
save(['.' filesep 'res_' file_signature filesep 'threshold_' file_signature]);

% Generation of the options for the policies
opt_policies = generate_opt_policies();
opt_policies.mu_max = mu_max;

%% EXPERIMENTS
load(['.' filesep 'res_' file_signature filesep 'threshold_' file_signature]);

for pp = 1 : n_policies
    fprintf('Policy %i of %i: %s...',pp,n_policies,names{pp});
    temp_pol = policies{pp};
    
    for cc = 1 : n_configurations
        temp_arms = configuration(cc).arms;

        reg = zeros(n_repetitions,n_iterations);
        best_arm = zeros(n_repetitions,n_iterations);
        for rr = 1:n_repetitions
            [best_arm(rr,:), reg(rr,:)] = ...
                run_configuration(thresholds(cc,rr,:), temp_arms, values{cc}, temp_pol, n_iterations, n_fix, opt_policies);
        end

        save(['.' filesep 'res_' file_signature filesep 'res_' names{pp} '_' num2str(cc) '_' file_signature],'-v7.3');
    end
    fprintf(' DONE\n');
end
