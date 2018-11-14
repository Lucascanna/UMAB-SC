# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:32:06 2018

@author: lucas
"""
import numpy as np
import time
import logging
import os
import pickle as pk

from Experiment import Experiment

class Executor:
    """
    This class wraps all the methods for executing policies on configurations.
    The regret is not computed here. It his computed a posteriori, when execution terminates.
    This is done to make the code as more modular as possible
    """
    
    def run_configuration(self, config, policies, num_repetitions, time_horizon):
        """
        Runs all the given policies on a given configuration.
        Each policy is run num_repetition times.
        Results are written on a pikle file
        """
        for ii in range(num_repetitions):
            path = 'results/'+ str(num_repetitions) + 'rep' + str(time_horizon) + 'times/'+ config.name + '/exp' + str(ii)
            
            #generate and save the realization
            config.gen_stochastic_processes()
            os.makedirs(os.path.dirname(path + '/config.pkl'), exist_ok=True)
            with open(path + '/config.pkl', 'wb') as f:
                pk.dump(config, f)
            
            #run each policy on the realization and save results
            for policy in policies:
                str_info='Executing experiment '+ str(ii) + ' of policy ' + policy.name + ' ...'
                logging.info(str_info)
                print(str_info)
                start_time = time.clock()
                
                experiment = Experiment(policy)
                pulls, rewards, switch_fees = self.run_policy(config, policy)
                experiment.set_results(pulls, rewards, switch_fees)
                experiment.save_results(path + '/' + policy.name + '.pkl')
                
                exp_time = time.clock() - start_time
                str_info='TIME TO PERFORM EXPERIMENT' + str(ii) + 'OF '+ policy.name + ' POLICY: '+ str(exp_time)
                logging.info(str_info)
                print(str_info)
    
    
    def run_policy(self, config, policy):
        """
        Runs the given policy on the given configuration
        """
        policy.reset_state(config.K, config.N)
        
        #switching costs are deterministic or stochastic depending on their shape
        isDeterministic = len(config.switch_costs.shape) == 2
        
        #for each arm, number of failures successes and pulls
        p0 = np.zeros(config.K, dtype=np.int32)
        p1 = np.zeros(config.K, dtype=np.int32)
        T = np.zeros(config.K, dtype=np.int32)
        
        # for each pair of arms, the number of times we payed/not payed for switching
        # if config is deterministic s0 is None and s1 contains the true costs
        if isDeterministic:
            s0 = None
            s1 = config.switch_costs
        else:
            s0 = np.zeros((config.K, config.K), dtype=np.int32)
            np.fill_diagonal(s0, 1)
            s1 = np.zeros((config.K, config.K), dtype=np.int32)
        
        #arm pulled for each time istant
        pulled = np.empty(config.N, dtype=np.int32)
        
        #reward at each time instant
        rewards = np.empty(config.N, dtype=np.int32)
        #switch fee at each time instant
        switch_fees = np.empty(config.N, dtype=np.int32)
        switch_fees[0] = 0

        
        #execution
        for n in range(config.N):
            pulled[n] = policy.choose_arm(n, p0, p1, s0, s1)
            
            rewards[n] = config.X[T[pulled[n]], pulled[n]]
            
            if n>0:
                # distinguish between deterministic and stochastic switching costs
                if isDeterministic:
                    switch_fees[n] = config.switch_costs[pulled[n-1], pulled[n]]
                else:
                    switch_fees[n] = config.switch_costs[n, pulled[n-1], pulled[n]]
                    # update s0 and s1 only if config is stochastic
                    if switch_fees[n] == 1:
                        s1[pulled[n-1], pulled[n]] += 1
                    else:
                        s0[pulled[n-1], pulled[n]] += 1
            
            #update p0 and p1
            T[pulled[n]] += 1
            if rewards[n] == 1:
                p1[pulled[n]] += 1
            else:
                p0[pulled[n]] += 1
            
        return pulled, rewards, switch_fees    
    
        
    
    