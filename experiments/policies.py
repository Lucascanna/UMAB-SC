# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:02:18 2018

@author: lucas
"""

from abc import ABC, abstractmethod
from math import e, log, ceil, pi
from scipy.optimize import newton
from regrets import SamplingRegret, SwitchingRegret, TotalRegret
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
    
    @abstractmethod
    def theoretical_bound(self, config, regret=None):
        """
        Computes the theoretical bound of the sampling regret
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
    
    
    def theoretical_bound(self, config, regret=None):
        n = np.arange(config.N)+1
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        delta = config.mu_max - suboptimal_mu
        
        if type(regret) == SamplingRegret:            
            bound = 8*np.log(n)*np.sum(np.true_divide(1, delta)) + (1 + (pi**2)/3)*np.sum(delta)
            return bound
        
        if type(regret) == SwitchingRegret:
            bound = 2*config.K*(1 + (pi**2)/3) + 16*np.log(n)*np.sum(np.true_divide(1, delta**2))
            return bound
        
        if type(regret) == TotalRegret:
            bound = 8*np.log(n)*np.sum(np.true_divide(1, delta)) + (1 + (pi**2)/3)*np.sum(delta) + 2*config.K*(1 + (pi**2)/3) + 16*np.log(n)*np.sum(np.true_divide(1, delta**2))
            return bound
        
        print("Error! Default implementation executed!")
        
    
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
        
        # init phase
        if n < self.K:
            self.current_arm=n
            return n
        
        # loop phase
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:
            if n > self.K:
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
    
    def theoretical_bound(self, config, regret=None):
        n = np.arange(config.N)+1
        
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        delta = config.mu_max - suboptimal_mu
        
        c_alpha = 1 + ((1 + self.alpha)*e)/(self.alpha**2) + (((1+self.alpha)/self.alpha)**(1+self.alpha))*(1+(11*(1+self.alpha))/(5*(self.alpha**2)*log(1+self.alpha)))
        
        if type(regret) == SamplingRegret:            
            bound = 0
            for delta_i in delta:
                bound = bound + (((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*delta_i) + c_alpha/delta_i)
                return bound
        
        if type(regret) == SwitchingRegret:
            bound=0
            for delta_i in delta:
                bound = bound + np.true_divide(np.log(((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*(delta_i**2))+ c_alpha/(delta_i**2) + 1), np.log(1+self.alpha)) 
            return bound*2
        
        if type(regret) == TotalRegret:
            bound_samp = 0
            bound_sw = 0
            for delta_i in delta:
                bound_samp = bound_samp + (((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*delta_i) + c_alpha/delta_i)
                bound_sw = bound_sw + np.true_divide(np.log(((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*(delta_i**2))+ c_alpha/(delta_i**2) + 1), np.log(1+self.alpha))
            return bound_samp + bound_sw
        
        print("Error! Default implementation executed!")
        

#%%
    
class KLUCB_Policy(Policy):
    """
    It chooses the next arm to play, according to KL-UCB policy
    """
    
    def __init__(self):
        super().__init__()
        self.name='KLUCB'
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        # init phase
        if n < self.K:
            return n
        
        # loop phase
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        
        b = np.empty(self.K)
        for ii in range(self.K):
            b[ii] = newton(self.__kl_func, x0=0.5, fprime=self.__kl_fprime, args=(T[ii], mu_hat[ii], n))
        u = mu_hat + b
        
        return np.argmax(u)
    
    def __kl_func(self, q, T, p, n):
        return T*(p*log(p/q) + (1-p)*log((1-p)/(1-q))) - log(n)
    
    def __kl_fprime(self, q, T, p, n):
        return -(T*p)/q + (T*(1-p))/(1-q)
    
    
    def reset_state(self, K):
        self.K = K
    
    
    def theoretical_bound(self, config, regret=None):
        pass
    

#%%
    
class UCYCLE_Policy(Policy):
    """
    It chooses the next arm to play, according to UCYCLE policy
    """
    
    def __init__(self):
        super().__init__()
        self.name='UCYCLE'
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            self.current_arm=n
            return n
        
        # loop phase
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:
            
            T = p0 + p1
            mu_hat = np.true_divide(p1, T)
            b = np.true_divide(np.log(np.true_divide(self.K * n**4, self.delta)),
                                2*T) ** 0.5
            u = mu_hat + b
            
            #update state
            self.current_arm = np.argmax(u)
            self.epoch_steps = 1
            self.epoch_len = T[self.current_arm]
            
            return self.current_arm
    
    
    def reset_state(self, K):
        self.K=K
        self.delta=0.05
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
    
    
    def theoretical_bound(self, config, regret=None):
#        n = np.arange(config.N)+1
#        arms = np.arange(config.K)
#        suboptimal_arms = np.delete(arms, config.best_arm)
#        suboptimal_mu = config.mu[suboptimal_arms]
#        delta = config.mu_max - suboptimal_mu
#        delta_min = np.min(delta)
#        
#        if type(regret) == SamplingRegret:            
#            bound = np.true_divide(48*config.K*np.log(np.true_divide(config.K*(n**4), self.delta)), delta_min) + (10/3)*self.delta*config.K*np.log(np.true_divide(2*n, config.K))
#            return bound
#        
#        if type(regret) == SwitchingRegret:
#            bound = config.K*np.log(np.true_divide(2*n, config.K))
#            return bound
#        
#        if type(regret) == TotalRegret:
#            bound = np.true_divide(48*config.K*np.log(np.true_divide(config.K*(n**4), self.delta)), delta_min) + (10/3)*self.delta*config.K*np.log(np.true_divide(2*n, config.K)) + config.K*np.log(np.true_divide(2*n, config.K))
#            return bound
#        
#        print("Error! Default implementation executed!")
        pass
    

#%%
    
class TS_Policy(Policy):
    """
    It chooses the next arm to play, according to Thompson Sampling policy
    """
    
    def __init__(self):
        super().__init__()
        self.name='TS'
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        samples = np.random.beta(p1 + 1, p0 + 1)
        return np.argmax(samples)
    
    
    def reset_state(self, K):
        self.K = K
        
    
    
    def theoretical_bound(self, config, regret=None):
        pass

#%%


class TS2_Policy(Policy):
    """
    It chooses the next arm to play, according to Thompson Sampling policy in epochs
    """
    
    def __init__(self):
        super().__init__()
        self.name='TS2'
        self.alpha = 0.8
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:                
                
            samples = np.random.beta(p1 + 1, p0 + 1)
            
            #update state
            self.current_arm = np.argmax(samples)
            self.epoch_steps = 1
            self.epoch_len = self.__tau(self.epochs_arm[self.current_arm]+1) - self.__tau(self.epochs_arm[self.current_arm])
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm

        
    def __tau(self, r):
        return np.ceil(np.power(1+self.alpha, r))
    
    
    def reset_state(self, K):
        self.K = K
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.zeros(K, dtype=np.int32)
    
    
    def theoretical_bound(self, config, regret=None):
        pass

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
            #s = s0 + s1
            gamma_hat = np.true_divide(s1, s0 + s1)[self.current_arm,:]
        b = (np.true_divide(2*log(n), T))**0.5
        u = mu_hat + 2*b - gamma_hat
        
        self.current_arm = np.argmax(u)
        return self.current_arm
    
    
    def reset_state(self, K):
        self.K = K
        self.init_epoch = 0
        self.init_time = 0
        self.end_init_phase = K**2 - 1
        self.current_arm = -1
    
    def theoretical_bound(self, config, regret=None):
        pass