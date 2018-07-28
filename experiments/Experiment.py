# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:30:43 2018

@author: lucas
"""
import pickle as pk

class Experiment(object):
    
    def __init__(self, config):
        super(Experiment, self).__init__()
        self.config = config
        self.config.gen_stochastic_processes()
        self.pulls = {}
        self.reward = {}
        self.swith_fees = {}
        
        
    def set_results(self, pulls, reward, switch_fees, policy_name):
        self.pulls[policy_name] = pulls
        self.reward[policy_name] = reward
        self.swith_fees[policy_name] = switch_fees
        
    def save_results(self, filepath):
        pk.dump(self, open(filepath, 'wb'))