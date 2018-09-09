# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:32:05 2018

@author: lucas
"""
import numpy as np
import time
import os

from DataLoader import DataLoader
from Setting import Setting
from Plotter import Plotter    
    
def plot_configuration(config_name):
    
    #set the regrets
    setting = Setting()
    regrets = setting.set_regrets()
    num_regrets = len(regrets)
    
    #load experiments
    loader = DataLoader()  
    experiments = loader.load_config_experiments(config_name)
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
    
    #compute theoretical bounds
    th_bounds = np.zeros((num_policies, num_regrets, N))
    for pol_idx in range(num_policies):
        for reg_idx in range(num_regrets):
            th_bounds[pol_idx, reg_idx, :] = policies[pol_idx].theoretical_bound(config, regrets[reg_idx])
    
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
    plt.plot_regret(mean_regret, error, th_bounds, titles, labels, filename[:-1])
    
    
    
def main():
    
    for dir_name in os.listdir('results'):
        print('Regret of configuration: ', dir_name)
        start_time = time.clock()
       
        plot_configuration(dir_name)
        
        exp_time = time.clock() - start_time
        print('TIME TO COMPUTE AND PLOT REGRET OF ', dir_name , 'CONFIGURATION: ', exp_time)
        
    
if __name__ == "__main__":
    main()
    
        
                
        
