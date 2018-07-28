# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:02:18 2018

@author: lucas
"""

from abc import ABC, abstractmethod
from math import e, log, ceil
import numpy as np

#%%

class Policy(ABC):
    """
    The abstract class representing a generic policy.
    All the actual policies will have to implement the method that chooses the next arm to play 
    """
    
    def __init__(self):
        super().__init__()
    
    
    @abstractmethod
    def choose_arm(self, n, p0, p1, s0, s1):
        """
        Input:
            n: current time step
            p0: vector containing the number of failures of each arm
            p1: vector containing the number of successes of each arm
        
        Output:
            index of next chosen arm
        """
        pass
    
    @abstractmethod
    def reset_state(self):
        """
        Resets the state of the policy for a new iteration
        """
        pass
    
    
    

#%%
        
class UCB1_Policy(Policy):
    """
    It chooses the next arm to play, according to UCB1 policy
    """
    
    def __init__(self):
        super().__init__()
        self.name='UCB1'
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            return n
        
        # loop phase
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        b = (np.true_divide(2*log(n), T))**0.5
        u = mu_hat + b
        
        return np.argmax(u)
    
    
    def reset_state(self, K):
        self.K = K
    
#%%
        
class UCB2_Policy(Policy):
    """
    It chooses the next arm to play, according to UCB2 policy
    """
    
    def __init__(self):
        super().__init__()
        self.name='UCB2'
        self.alpha = 0.8
        
        
        
    def reset_state(self, K):
        self.K = K
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.zeros(K, dtype=np.int32)
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        K = p0.shape[0]
        
        # init phase
        if n < K:
            self.current_arm=n
            return n
        
        # loop phase
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:
            if n > K:
                self.epochs_arm[self.current_arm] += 1
            T = p0 + p1
            mu_hat = np.true_divide(p1, T)
            b = np.true_divide((1+self.alpha) * np.log(np.true_divide((e*n), self.__tau(self.epochs_arm))),
                                2*self.__tau(self.epochs_arm)) ** 0.5
            u = mu_hat + b
            
            #update state
            self.current_arm = np.argmax(u)
            self.epoch_steps = 1
            self.epoch_len = self.__tau(self.epochs_arm[self.current_arm]+1) - self.__tau(self.epochs_arm[self.current_arm])
            
            return self.current_arm
        
    def __tau(self, r):
        return np.ceil(np.power(1+self.alpha, r))

#%%
        
class UCB1SC_Policy(Policy):
    """
    It chooses the next arm to play, according to UCB1 policy.
    The index also takes into account cost for switching
    """
    
    def __init__(self):
        super().__init__()
        self.name='UCB1SC'
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        
        # init phase
        if n < self.end_init_phase:  
            
            if self.init_time > 2*(self.K - self.init_epoch - 1):
                self.init_epoch += 1
                self.init_time = 1
                self.current_arm = self.init_epoch
            else:
                if self.init_time % 2 == 0:
                    self.current_arm = self.init_epoch
                else:
                    self.current_arm = ceil(self.init_time/2) + self.init_epoch
                
                self.init_time += 1
            return self.current_arm
        
        # loop phase
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        if s0 is None:
            gamma_hat = s1[self.current_arm,:]
        else:
            s = s0 + s1
            gamma_hat = np.true_divide(s1, s0 + s1)[self.current_arm,:]
        b = (np.true_divide(2*log(n), T))**0.5
        u = mu_hat + b - gamma_hat
        
        self.current_arm = np.argmax(u)
        return self.current_arm
    
    
    def reset_state(self, K):
        self.K = K
        self.init_epoch = 0
        self.init_time = 0
        self.end_init_phase = K**2 - 1
        self.current_arm = -1