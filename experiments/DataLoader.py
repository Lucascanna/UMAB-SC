# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:24:38 2018

@author: lucas
"""
import pickle as pk
import os

class DataLoader():
    
    def load_config_experiments(self, config_name):
    
        experiments = {}
        for policy_name in os.listdir('results/'+config_name):
            experiments[policy_name]=[]
            for filename in os.listdir('results/'+config_name+'/'+policy_name):
                exp = pk.load(open('results/'+config_name+'/'+policy_name+'/'+filename, 'rb'))
                experiments[policy_name].append(exp)
        return experiments
        