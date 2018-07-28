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

def compute_plot_regret(config_name):
    
    setting = Setting()
    
    regrets = setting.set_regrets()
    num_regrets = len(regrets)
    
    loader = DataLoader()
        
    experiments = loader.load_experiments('results/'+config_name)
    num_exp = len(experiments)
    
    N = experiments[0].config.N
    policies = list(experiments[0].pulls.keys())
    num_policies = len(policies)

    
    regret = np.zeros((num_policies, num_regrets, N))
    for pol_idx in range(num_policies):
        curr_policy = policies[pol_idx]
        for exp_idx in range(num_exp):
            curr_exp = experiments[exp_idx]
            pulls = curr_exp.pulls[curr_policy]
            for reg_idx in range(num_regrets):
                curr_regr = regrets[reg_idx]
                regr = curr_regr.compute_regret(pulls, curr_exp.config)
                regret[pol_idx, reg_idx, :] = regret[pol_idx, reg_idx, :] + np.true_divide(regr, num_exp)
            
    
    #plot the regrets
    titles = []
    filename = config_name + '_'
    for policy in policies:
        filename = filename + policy + '-'
    filename = filename[:-1] + '_'
    for reg_type in regrets:
        titles = titles + [reg_type.description]
        filename = filename + reg_type.name + '-'
    
    plt = Plotter()
    plt.plot_regret(regret, titles, policies, filename[:-1])
    
    
    
def main():
    
    for dir_name in os.listdir('results'):
        print('Regret of configuration: ', dir_name)
        start_time = time.clock()
        
        compute_plot_regret(dir_name)
        
        exp_time = time.clock() - start_time
        print('TIME TO COMPUTE AND PLOT REGRET OF ', dir_name , 'CONFIGURATION: ', exp_time)
        
    
if __name__ == "__main__":
    main()
    
        
                
        
