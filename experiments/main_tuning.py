# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:12:44 2018

@author: lucas
"""
import time
import numpy as np

from policies import UCB2_Policy
from configurations import MUUD_Configuration
from Executor import Executor
from DataLoader import DataLoader
from Setting import Setting
from Plotter import Plotter

def plot_configuration(config_name, policy_names):
    
    #set the regrets
    setting = Setting()
    regrets = setting.set_regrets()
    num_regrets = len(regrets)
    
    #load experiments
    loader = DataLoader()  
    experiments = loader.load_config_experiments(config_name, policy_names)
    num_exp = len(list(experiments.values())[0])
    
    #retrieve configuration
    config = list(experiments.values())[0][0].config
    
    #retrive time horizon
    N = config.N
    
    #retrive policies
    policies=[]
    for exp_list in experiments.values():
        policies = policies + [exp_list[0].policy]
    num_policies = len(policies)

    #compute the regrets
    regret = np.zeros((num_exp, num_policies, num_regrets, N))
    for pol_idx in range(num_policies):
        curr_policy = policies[pol_idx]
        for exp_idx in range(num_exp):
            curr_exp = experiments[curr_policy.name][exp_idx]
            pulls = curr_exp.pulls
            for reg_idx in range(num_regrets):
                curr_regr = regrets[reg_idx]
                regret[exp_idx, pol_idx, reg_idx, :] = curr_regr.compute_regret(pulls, curr_exp.config)
                
    mean_regret = np.mean(regret, axis=0)
    
    #varianza del regret
    std_regret = np.std(regret, axis=0)
    error = 2*np.true_divide(std_regret, num_exp**0.5)
    
    #plot the regrets
    titles = []
    labels = []
    filename = config_name + '_'
    for policy in policies:
        filename = filename + policy.name + '-'
        labels = labels + [policy.name]
    filename = filename[:-1] + '_'
    for reg_type in regrets:
        titles = titles + [reg_type.description]
        filename = filename + reg_type.name + '-'
    
    plt = Plotter()
    plt.plot_regret(mean_regret, error, None, titles, labels, filename[:-1])

def main():
    
    #set configuration
    N = 2**17
    num_repetitions=5
    config = MUUD_Configuration(N)
    
    #set possible values of a parameter
    values = np.arange(0.1, 1, 0.1)
    
    #set policy
    policies = []
    policy_names = []
    for value in values:
        policy = UCB2_Policy(alpha=value)
        policy.name = policy.name + str(value)
        policies = policies + [policy]
        policy_names = policy_names + [policy.name]
        
    #execute experiments
    exe = Executor()
    print('Executing experiments for tunining...')
    start_time = time.clock()
    
    exe.run_configuration(config, policies, num_repetitions)
    
    exp_time = time.clock() - start_time
    print('TIME TO PERFORM EXPERIMENTS: ', exp_time)
    
    
    #printing the results
    print('Printing tuning results...')
    start_time = time.clock()
    
    plot_configuration(config.name, policy_names)
    
    exp_time = time.clock() - start_time
    print('TIME TO PRINT RESULTS: ', exp_time)
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()