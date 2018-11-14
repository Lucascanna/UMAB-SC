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
    
        experiments = {}
        for policy_name in policy_names:
            experiments[policy_name]=[]
            for ii in range(num_rep):
                with open(path+'/exp' + str(ii) + '/' + policy_name + '.pkl', 'rb') as f:
                    exp = pk.load(f)  
                experiments[policy_name].append(exp)
        return experiments
    
    
    def load_realizations(self, config_name, num_rep, N):
        path = 'results/'+str(num_rep)+'rep'+str(N)+'times/'+config_name
        reals = []
        for ii in range(num_rep):
            with open(path+'/exp'+str(ii)+'/config.pkl', 'rb') as f:
                real = pk.load(f)
            reals.append(real)
            
        return reals
        
        