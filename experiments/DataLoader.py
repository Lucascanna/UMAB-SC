# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:24:38 2018

@author: lucas
"""
import pickle as pk
import os

class DataLoader():
    
    def load_config_experiments(self, config_name, policy_names, num_rep, N):
        path = 'results/'+str(num_rep)+'rep'+str(N)+'times/'+config_name
        if not set(policy_names).issubset(os.listdir(path)):
            raise LookupError("There are no experiments for this setting")
    
        experiments = {}
        for policy_name in policy_names:
            experiments[policy_name]=[]
            for filename in os.listdir(path+'/'+policy_name):
                exp = pk.load(open(path+'/'+policy_name+'/'+filename, 'rb'))
                experiments[policy_name].append(exp)
        return experiments
        