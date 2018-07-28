# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:15:15 2018

@author: lucas
"""

from policies import UCB1_Policy, UCB2_Policy, UCB1SC_Policy
from regrets import SamplingRegret, SwitchingRegret, TotalRegret
from configurations import MUUD_Configuration, MUUS_Configuration

class Setting():
    """
    In this class, the user can choose policies, regrets and configurations of the experiment.
    Just comment/uncomment policies, regrets and configurationa you want to exclude/include in the experiment.
    """
    
    def set_policies(self):
        policies = [
                    UCB1_Policy(), 
                    UCB2_Policy(),
                    UCB1SC_Policy()
                    ]
        return policies
    
    
    def set_configurations(self, N):
        configs = [
                    MUUD_Configuration(N),
                    MUUS_Configuration(N)
                    ]
        return configs
    
    
    def set_regrets(self):
        regrets = [
                    SamplingRegret(),
                    SwitchingRegret(),
                    TotalRegret()
                    ]
        
        return regrets
                
                
            