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
    def compute_regret(self, config, pulls, true_rewards, true_switch_fees):
        pass
        
        
        
        
#%%
        
class SamplingRegret(Regret):
    """
    Pseudo-regret assuming to know the best arm:
        E[R] = mu_best * N - mu_i * E[T_i]
    """
    def __init__(self):
        super().__init__()
        self.name='samp'
        self.description='Sampling Regret'
        
    
    def compute_regret(self, config, pulls, true_rewards, true_switch_fees):
        mu_best = np.max(config.mu)
        
        pseudo_rewards = config.mu[pulls]
        
        regret = mu_best - pseudo_rewards
        
        cum_regret = np.cumsum(regret)
        
        return cum_regret


#%%
        
class SwitchingRegret(Regret):
    """
    Pseudo-Regret caused by the switching among arms:
        R = gamma_i,j * S_i,j
    """
    
    def __init__(self):
        super().__init__()
        self.name='sw'
        self.description='Switching Regret'
    
    def compute_regret(self, config, pulls, true_rewards, true_switch_fees):
        from_vect = pulls[:-1]
        to_vect = pulls[1:]
        
        regret = config.gamma[from_vect, to_vect]
        
        cum_regret = np.cumsum(regret)
        cum_regret = np.insert(cum_regret, 0, 0)
        
        return cum_regret

#%%
        
class TotalRegret(Regret):
    """
    The sum of sampling and switching regret
    """
    
    def __init__(self):
        super().__init__()
        self.name='tot'
        self.description='Total Regret'
    
    def compute_regret(self, config, pulls, true_rewards, true_switch_fees):
        samp_regret = SamplingRegret()
        samp_regret_values = samp_regret.compute_regret(config, pulls, true_rewards, true_switch_fees)
        
        sw_regret = SwitchingRegret()
        sw_regret_values = sw_regret.compute_regret(config, pulls, true_rewards, true_switch_fees)
        
        return samp_regret_values + sw_regret_values
    
#%%
    
class BudgetRegret(Regret):
    """
    The pseudo-regret considering budget on switching costs
    """
    def __init__(self, B):
        super().__init__()
        self.name='bdg'
        self.description='Budget Regret'
        self.B = B
    
    def compute_regret(self, config, pulls, true_rewards, true_switch_fees):
        mu_best = np.max(config.mu) 
        pseudo_rewards = config.mu[pulls]
        
        from_vect = pulls[:-1]
        to_vect = pulls[1:]
        pseudo_costs = config.gamma[from_vect, to_vect]
        pseudo_costs = np.insert(pseudo_costs,0,0)
        cum_costs = np.cumsum(pseudo_costs)
        n_hat = np.argmax(cum_costs > self.B)
        
        #soluzione 1
        if not n_hat == 0:
            pseudo_rewards[n_hat:] = 0
        regret = mu_best - pseudo_rewards
        cum_regret = np.cumsum(regret)
        
        return cum_regret
    
#%%
        
class BudgetedReward(Regret):
    """
    The pseudo-reward constrained by the budget on switching costs
    """
    def __init__(self, B):
        super().__init__()
        self.name = 'Brew'
        self.description='Budgeted Reward'
        self.B=B
    
    def compute_regret(self, config, pulls, true_rewards, true_switch_fees):
        pseudo_rewards = config.mu[pulls]
        
        from_vect = pulls[:-1]
        to_vect = pulls[1:]
        pseudo_costs = config.gamma[from_vect, to_vect]
        pseudo_costs = np.insert(pseudo_costs,0,0)
        cum_costs = np.cumsum(pseudo_costs)
        n_hat = np.argmax(cum_costs > self.B)
        
        if not n_hat == 0:
            pseudo_rewards[n_hat:] = 0
        cum_reward = np.cumsum(pseudo_rewards)
        
        return cum_reward
        
        