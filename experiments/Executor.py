# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:32:06 2018

@author: lucas
"""
import numpy as np

from Experiment import Experiment

class Executor:
    """
    This class wraps all the methods for executing policies on configurations.
    The regret is not computed here. It his computed a posteriori, when execution terminates.
    This is done to make the code as more modular as possible
    """
    
    def run_configuration(self, config, policies, num_repetitions):
        """
        Runs all the given policies on a given configuration.
        Each policy is run num_repetition times.
        Results are written on a pikle file
        """
        for ii in range(num_repetitions):
            experiment = Experiment(config)
            #execute policies
            for policy in policies:
                pulls, rewards, switch_fees = self.run_policy(experiment.config, policy)
                experiment.set_results(pulls, rewards, switch_fees, policy.name)
            experiment.save_results('results/'+ config.name + '/exp' + str(ii) + '.pkl')

    
    
    def run_policy(self, config, policy):
        """
        Runs the given policy on the given configuration
        """
        policy.reset_state(config.K)
        
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
    
    
    
        
    
    