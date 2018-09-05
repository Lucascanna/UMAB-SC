# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:30:43 2018

@author: lucas
"""
import pickle as pk
import os

class Experiment(object):
    """
    An Experiment is the execution of several policies on one instance of a specific configuration
    """
    
    def __init__(self, config, policy):
        super(Experiment, self).__init__()
        self.config = config
        self.policy = policy
        self.config.gen_stochastic_processes()
        
        
    def set_results(self, pulls, reward, switch_fees):
        self.pulls = pulls
        self.reward = reward
        self.swith_fees = switch_fees
        
    def save_results(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True) 
        pk.dump(self, open(filepath, 'wb'))