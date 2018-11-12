# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:26:23 2018

@author: lucas
"""

import numpy as np
from abc import ABC, abstractmethod

#%%

class Configuration(ABC):
    """
    Abstract configuration of a MAB-SC problem.
    Each configuration distingishes from:
        - number of arms
        - the generation of true mean rewards
        - the generation of true mean switching costs
        - stochastic or deterministic switching costs
    """
    
    def __init__(self, N):
        super().__init__()
        self.N = N    #time horizon
    
    @abstractmethod
    def gen_stochastic_processes(self):
        """
        Generates all the values of the stochastic processes involved.
        """
        pass
    

#%%
        
class MUUD_Configuration(Configuration):
    
    
    def __init__(self, N):
        super().__init__(N)
        self.name = 'MUUD'
        self.description = 'MUUD: Many arms, Uniform true mean rewards, Uniform switching costs, Deterministic switching costs'
        #number of arms
        self.K = 16
        
        self.seed=10
        np.random.seed(self.seed)
        
        # true mean rewards
        self.mu = np.random.rand(self.K)
        self.best_arm = np.argmax(self.mu)
        self.mu_max = np.max(self.mu)
        
        # deterministic switching costs
        self.gamma = np.random.rand(self.K, self.K)
        np.fill_diagonal(self.gamma,0)
        self.gamma = (self.gamma + self.gamma.T)/2 #make the matrix symmetric
        self.switch_costs = self.gamma
    
    
    def gen_stochastic_processes(self):
        """
        Generates the values of all the rewards
        """
        self.seed += 1
        np.random.seed(self.seed)
        
        # bernoulli rewards of each arm at each time instant
        self.X = np.random.binomial(1, self.mu, size=(self.N,self.K))
        
        

#%%

class MUUS_Configuration(MUUD_Configuration):
    
    def __init__(self, N):
        super().__init__(N)
        self.name = 'MUUS'
        self.description = 'MUUS: Many arms, Uniform true mean rewards, Uniform switching costs, Stochastic switching costs'
        
    
    def gen_stochastic_processes(self):
        """
        Generates the values of both rewards and switching costs
        """
        
        # bernoulli rewards of each arm at each time instant
        super().gen_stochastic_processes()
        
        # stochastic switching costs
        self.switch_costs = np.random.binomial(1, self.gamma, size= (self.N, self.K, self.K))
        
#%%

class MCUD_Configuration(Configuration):
    
    def __init__(self, N):
        super().__init__(N)
        self.name = 'MCUD'
        self.description = 'MCUD: Many arms, Close true mean rewards, Uniform switching costs, Deterministic switching costs'
        
        #number of arms
        self.K = 16
        
        self.seed=10
        np.random.seed(self.seed)
        
        # true mean rewards
        self.mu = np.random.uniform(0.5, 1.0, self.K)
        self.best_arm = np.argmax(self.mu)
        self.mu_max = np.max(self.mu)
        
        # deterministic switching costs
        self.gamma = np.random.rand(self.K, self.K)
        np.fill_diagonal(self.gamma,0)
        self.gamma = (self.gamma + self.gamma.T)/2 #make the matrix symmetric
        self.switch_costs = self.gamma
        
    def gen_stochastic_processes(self):
        """
        Generates the values of all the rewards
        """
        self.seed += 1
        np.random.seed(self.seed)
        
        # bernoulli rewards of each arm at each time instant
        self.X = np.random.binomial(1, self.mu, size=(self.N,self.K))
        
                
                
            