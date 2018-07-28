% Script to plot the results of the experiments on different pricing configurations by
% considering different MAB policies
%
%   Copyright 2015 Paladino, S. and Trovo', F., Politecnico di Milano

clc
clear
close all
addpath(genpath('.'));

% Experimental setting
n_iterations = 10000;
n_repetitions = 2;
n_configurations = 3;
data = '24-May-2017';

file_signature = [num2str(n_configurations) '_' num2str(n_repetitions) '_' num2str(n_iterations) '_' data];
load(['.' filesep 'res_' file_signature filesep 'threshold_' file_signature]);

mean_regret = cell(n_configurations,n_policies);
std_regret = cell(n_configurations,n_policies);

% Mean and std extraction
for pp = 1 : n_policies
    fprintf('Evaluating mean and std of policy %i of %i: %s...',pp,n_policies,names{pp});
    for cc = 1 : n_configurations
        load(['.' filesep 'res_' file_signature filesep 'res_' names{pp} '_' num2str(cc) '_' file_signature]);
        
        mean_regret{cc,pp} = mean(reg);
        std_regret{cc,pp} = 2 * std(reg) / sqrt(n_repetitions);
    end
    fprintf(' DONE\n');
end

indexes = 1:n_policies;

% Plot
figu_r = plot_comparison(mean_regret(:,indexes), std_regret(:,indexes), names(indexes), configuration);
