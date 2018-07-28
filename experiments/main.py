# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:39:59 2018

@author: lucas
"""

import numpy as np
import time

from Executor import Executor
from Plotter import Plotter
from Setting import Setting


# run the experiment on a single configuration and produce a single picture
def run_experiment(config, setting, num_repetitions):
    
    #set policies
    policies = setting.set_policies(config.K)
    num_policies = len(policies)
    
    #set regrets
    regrets = setting.set_regrets()
    num_regrets = len(regrets)
    
    #execute policies and compute the regrets
    exe = Executor()
    curr_regret = np.zeros((num_policies, num_regrets, config.N))
    for pol_idx in range(num_policies):
        # execute each policy num_repetition times
        for ii in range(num_repetitions):
            pulls = exe.run_policy(config, policies[pol_idx])
            # regret update
            for reg_idx in range(num_regrets):
                regr = regrets[reg_idx].compute_regret(pulls, config)
                curr_regret[pol_idx, reg_idx, :] = curr_regret[pol_idx, reg_idx, :] + np.true_divide(regr, num_repetitions)
            # before starting a new repetition generate a new configuration and reset the state of the policy
            config.gen_stochastic_processes()
            policies[pol_idx].reset_state()   
    
    #plot the regrets
    titles = []
    labels = []
    styles = []
    filename = config.name + '_'
    for policy in policies:
        labels = labels + [policy.name]
        styles = styles + [policy.style]
        filename = filename + policy.name + '-'
    filename = filename[:-1] + '_'
    for regret in regrets:
        titles = titles + [regret.description]
        filename = filename + regret.name + '-'
    
    plt = Plotter()
    plt.plot_regret(curr_regret, titles, labels, styles, filename[:-1])


def main():
    
    #set configuration
    N = 2**17
    num_repetitions=5
    setting = Setting()
    configs = setting.set_configurations(N)
    
    # make an experiment for each configuration
    for config in configs:
        print('Experiment: ', config.name)
        start_time = time.clock()
        config.gen_stochastic_processes()
        run_experiment(config, setting, num_repetitions)
        exp_time = time.clock() - start_time
        print('TIME TO PERFORM EXPERIMENT ', config.name , 'CONFIGURATION: ', exp_time)

    

if __name__ == "__main__":
    main()