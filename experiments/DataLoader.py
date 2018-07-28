# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:24:38 2018

@author: lucas
"""
import pickle as pk
import os

class DataLoader():
    
    def load_experiments(self, path):
        experiments = []
        for filename in os.listdir(path):
            exp = pk.load(open(path + '/' + filename, 'rb'))
            experiments.append(exp)
        
        return experiments
        