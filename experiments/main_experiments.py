# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:10:42 2018

@author: lucas
"""

import time

from Executor import Executor
from Setting import Setting

#%%
    
def main():
    """
    All the experiments on all the configurations and all the policies have all the same true mean.
    The generation of the stochastic processes from this true means is different for each experiment.
    """
    
    #set configuration
    N = 2**17
    num_repetitions=5
    setting = Setting()
    configs = setting.set_configurations(N)
    
    #set policies
    policies = setting.set_policies()
    
    exe = Executor()
    

    for config in configs:
        print('Experiments of configuration: ', config.name)
        start_time = time.clock()
        
        exe.run_configuration(config, policies, num_repetitions)
        
        exp_time = time.clock() - start_time
        print('TIME TO PERFORM EXPERIMENT ', config.name , 'CONFIGURATION: ', exp_time)
    
    
        
if __name__ == "__main__":
    main()