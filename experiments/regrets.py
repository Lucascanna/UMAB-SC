# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:04:27 2018

@author: lucas
"""
import numpy as np
from abc import ABC, abstractmethod

#%%

class Regret(ABC):
    """
    Abstract class representing a generic regret.
    All the actual regrets will have to implement the method that computes the specific regret given the history of pulls and the configuration of the problem.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def compute_regret(self, pulls, config):
        pass
        
        
        
        
#%%
        
class SamplingRegret:
    """
    Pseudo-regret assuming to know the best arm:
        E[R] = mu_best * N - mu_i * E[T_i]
    """
    def __init__(self):
        super().__init__()
        self.name='samp'
        self.description='Sampling Regret'
        
    
    def compute_regret(self, pulls, config):
        mu_best = np.max(config.mu)
        
        rewards = config.mu[pulls]
        
        regret = mu_best - rewards
        
        cum_regret = np.cumsum(regret)
        
        return cum_regret


#%%
        
class SwitchingRegret():
    """
    Regret caused by the switching among arms:
        R = gamma_i,j * S_i,j
    """
    
    def __init__(self):
        super().__init__()
        self.name='sw'
        self.description='Switching Regret'
    
    def compute_regret(self, pulls, config):
        from_vect = pulls[:-1]
        to_vect = pulls[1:]
        
        # distinguish between deterministic and stochastic switching costs
        if len(config.switch_costs.shape) == 2:
            regret = config.switch_costs[from_vect, to_vect]
        else:
            n = np.arange(config.N)[1:]
            regret = config.switch_costs[n, from_vect, to_vect]
        
        cum_regret = np.cumsum(regret)
        cum_regret = np.insert(cum_regret, 0, 0)
        
        return cum_regret

#%%
        
class TotalRegret():
    """
    The sum of sampling and switching regret
    """
    
    def __init__(self):
        super().__init__()
        self.name='tot'
        self.description='Total Regret'
    
    def compute_regret(self, pulls, config):
        samp_regret = SamplingRegret()
        samp_regret_values = samp_regret.compute_regret(pulls, config)
        
        sw_regret = SwitchingRegret()
        sw_regret_values = sw_regret.compute_regret(pulls, config)
        
        return samp_regret_values + sw_regret_values
        
        
        