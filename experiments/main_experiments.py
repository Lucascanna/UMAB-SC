# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:10:42 2018

@author: lucas
"""

import time
import logging

from Executor import Executor
import settings as st

#%%

def main():
    """
    All the experiments on all the configurations and all the policies have all the same true mean.
    The generation of the stochastic processes from this true means is different for each experiment.
    """
    #log config
    logging.basicConfig(filename=st.LOG_FILENAME,level=logging.INFO)
    
    exe = Executor()
    for config in st.configs:
        str_info= '\n Executing experiments of configuration ' + config.name + ' with N=' + str(st.N) + ' and num_rep=' + str(st.num_repetitions) + ' ...'
        logging.info(str_info)
        print(str_info)
        start_time = time.clock()
        
        exe.run_configuration(config, st.policies, st.num_repetitions, st.N)
        
        exp_time = time.clock() - start_time
        str_info = 'TIME TO PERFORM EXPERIMENTS OF ' + config.name + 'CONFIGURATION: ' + str(exp_time)
        logging.info(str_info)
        print(str_info)
    
    
        
if __name__ == "__main__":
    main()