# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:32:05 2018

@author: lucas
"""
import numpy as np
import time
from math import ceil
import logging

from DataLoader import DataLoader
from Plotter import Plotter  
import settings as st
    
def plot_configuration(config_name, policy_names):
    
    #set the regrets
    num_regrets = len(st.regrets)
    
    #load experiments
    loader = DataLoader()  
    experiments = loader.load_config_experiments(config_name, policy_names, st.num_repetitions, st.N)
    num_exp = len(list(experiments.values())[0])
    
    #load the realizations
    reals = loader.load_realizations(config_name, st.num_repetitions, st.N)
    
    #retrive policies and their colors
    policies=[]
    colors = []
    err_colors = []
    for exp_list in experiments.values():
        policy = exp_list[0].policy
        policies = policies + [policy]
        colors = colors + [policy.color]
        err_colors = err_colors + [policy.err_color]
    num_policies = len(policies)

    #compute the regrets 
    regret = np.zeros((num_exp, num_policies, num_regrets, ceil(st.N/st.sample_rate)))
    for pol_idx in range(num_policies):
        curr_policy = policies[pol_idx]
        for exp_idx in range(num_exp):
            curr_exp = experiments[curr_policy.name][exp_idx]
            curr_real = reals[exp_idx]
            for reg_idx in range(num_regrets):
                curr_regr = st.regrets[reg_idx]
                regret[exp_idx, pol_idx, reg_idx, :] = curr_regr.compute_regret(curr_real, curr_exp.pulls, curr_exp.reward, curr_exp.swith_fees)[::st.sample_rate]
                
    mean_regret = np.mean(regret, axis=0)
    n = np.arange(st.N)[::st.sample_rate]
    
    #varianza del regret
    std_regret = np.std(regret, axis=0)
    error = 2*np.true_divide(std_regret, num_exp**0.5)
    
    #compute theoretical bounds
    th_bounds = None
    if st.plot_th_bounds:
        th_bounds = np.zeros((num_policies, num_regrets, ceil(st.N/st.sample_rate)))
        for pol_idx in range(num_policies):
            for reg_idx in range(num_regrets):
                th_bounds[pol_idx, reg_idx, :] = policies[pol_idx].theoretical_bound(reals[0], st.regrets[reg_idx])[::st.sample_rate]
    
    #plot the regrets
    titles = []
    labels = []
    filename = config_name + '_'
    for policy in policies:
        filename = filename + policy.name + '-'
        labels = labels + [policy.name]
    filename = filename[:-1] + '_'
    for reg_type in st.regrets:
        titles = titles + [reg_type.description]
        filename = filename + reg_type.name + '-'
    
    plt = Plotter()
    plt.plot_regret(mean_regret, n, error, colors, err_colors, th_bounds, titles, labels, filename[:-1])
    
    
    
def main():
    
    config_names = [config.name for config in st.configs]        
        
    policies_name = [policy.name for policy in st.policies]
    
    #log config
    logging.basicConfig(filename=st.LOG_FILENAME,level=logging.INFO)
    
    for config_name in config_names:
        str_info = '\n Computing and plotting regrets of configuration: ' + config_name + ' with N=' + str(st.N) + ';  num_rep=' + str(st.num_repetitions) + 'sample_rate=' + str(st.sample_rate) + ' ...'
        print(str_info)
        logging.info(str_info)
        start_time = time.clock()
       
        plot_configuration(config_name, policies_name)
        
        exp_time = time.clock() - start_time
        str_info= 'TIME TO COMPUTE AND PLOT REGRET OF ' + config_name + 'CONFIGURATION: ' +  str(exp_time)
        print(str_info)
        logging.info(str_info)
        
    
if __name__ == "__main__":
    main()
    
        
                
        
