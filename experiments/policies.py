# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:02:18 2018

@author: lucas
"""

from abc import ABC, abstractmethod
from math import e, log, ceil, pi
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
    
    def kl(self, p, q):
        """
        The KL-divergence of Bernoulli distributions
        """
        eps = 1e-15  # Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps] 
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)
        return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
    
    
    

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
            #con costanti additive
            #bound = 8*np.log(n)*np.sum(np.true_divide(1, delta)) + (1 + (pi**2)/3)*np.sum(delta)
            
            #senza costanti additive
            bound = 8*np.log(n)*np.sum(np.true_divide(1, delta))
            
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        if type(regret) == SwitchingRegret:
            #con costanti additive
            #bound = 2*config.K*(1 + (pi**2)/3) + 16*np.log(n)*np.sum(np.true_divide(1, delta**2))
            
            #senza costanti additive
            bound = 16*np.log(n)*np.sum(np.true_divide(1, delta**2))
            
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        if type(regret) == TotalRegret:
            #con costanti additive
            #bound = 8*np.log(n)*np.sum(np.true_divide(1, delta)) + (1 + (pi**2)/3)*np.sum(delta) + 2*config.K*(1 + (pi**2)/3) + 16*np.log(n)*np.sum(np.true_divide(1, delta**2))
            
            #senza costanti additive
            bound = 8*np.log(n)*np.sum(np.true_divide(1, delta)) + 16*np.log(n)*np.sum(np.true_divide(1, delta**2))
            
            bound = np.insert(bound[:-1],0,0)
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
        
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        delta = config.mu_max - suboptimal_mu
        delta_min = np.min(delta)
        start = 1/(2*(delta_min**2))
        n = np.arange(config.N)+ceil(start)
        
        initial_regret = np.arange(ceil(start))
        
        #c_alpha = 1 + ((1 + self.alpha)*e)/(self.alpha**2) + (((1+self.alpha)/self.alpha)**(1+self.alpha))*(1+(11*(1+self.alpha))/(5*(self.alpha**2)*log(1+self.alpha)))
        
        if type(regret) == SamplingRegret:            
            bound = 0
            for delta_i in delta:
                #con costanti additive
                #bound = bound + (((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*delta_i) + c_alpha/delta_i)
                
                # senza costanti additive
                bound = bound + (((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*delta_i))
            
            bound = np.insert(bound[:-ceil(start)],0,initial_regret)
            return bound
        
        if type(regret) == SwitchingRegret:
            bound=0
            for delta_i in delta:
                #con costanti additive
                #bound = bound + np.true_divide(np.log(((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*(delta_i**2))+ c_alpha/(delta_i**2) + 1), np.log(1+self.alpha))
                
                #senza costanti additive
                bound = bound + np.true_divide(np.log(((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*(delta_i**2))), np.log(1+self.alpha))
                
            bound = np.insert(bound[:-ceil(start)],0,initial_regret)    
            return bound*2
        
        if type(regret) == TotalRegret:
            bound_samp = 0
            bound_sw = 0
            for delta_i in delta:
                #con costanti additive
                #bound_samp = bound_samp + (((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*delta_i) + c_alpha/delta_i)
                #bound_sw = bound_sw + np.true_divide(np.log(((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*(delta_i**2))+ c_alpha/(delta_i**2) + 1), np.log(1+self.alpha))
                
                #senza costanti additive
                bound_samp = bound_samp + (((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*delta_i))
                bound_sw = bound_sw + np.true_divide(np.log(((1+self.alpha)*(1 + 4*self.alpha)*np.log(2*e*n*(delta_i**2)))/(2*(delta_i**2))), np.log(1+self.alpha))
                
            bound = bound_samp + bound_sw
            bound = np.insert(bound[:-ceil(start)],0,initial_regret)
            return bound
        
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
        
        compute_indexes = np.vectorize(self.__compute_arm_index)
        q = compute_indexes(mu_hat, np.log(n)/T)
        
        return np.argmax(q)
    
    def __compute_arm_index(self, x, d, upperbound=1, lowerbound=0):
        precision=1e-6
        max_iterations=50
        
        value = max(x, lowerbound)
        u = upperbound
        
        _count_iteration = 0
        while _count_iteration < max_iterations and u - value > precision:
            _count_iteration += 1
            m = (value + u) / 2.
            if self.kl(x, m) > d:
                u = m
            else:
                value = m
                
        return (value + u) / 2
        
    
    def reset_state(self, K):
        self.K = K
    
    
    def theoretical_bound(self, config, regret=None):
        n = np.arange(config.N)+2
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        delta = config.mu_max - suboptimal_mu
        
        kl_vect = np.vectorize(self.kl)
        
        if type(regret) == SamplingRegret:            
    
            #senza costanti additive
            bound = np.log(n)*np.sum(np.true_divide(delta, kl_vect(suboptimal_mu, config.mu_max))) + np.log(np.log(n))
            
            bound = np.insert(bound[:-2],0,[0,0])
            return bound
        
        if type(regret) == SwitchingRegret:
            
            #senza costanti additive
            bound = 2*np.log(n)*np.sum(np.true_divide(1, kl_vect(suboptimal_mu, config.mu_max))) + 2*config.K*np.log(np.log(n))
            
            bound = np.insert(bound[:-2],0,[0,0])
            return bound
        
        if type(regret) == TotalRegret:
        
            #senza costanti additive
            bound = np.log(n)*np.sum(np.true_divide(delta, kl_vect(suboptimal_mu, config.mu_max))) + np.log(np.log(n)) + 2*np.log(n)*np.sum(np.true_divide(1, kl_vect(suboptimal_mu, config.mu_max))) + 2*config.K*np.log(np.log(n))
            
            bound = np.insert(bound[:-2],0,[0,0])
            return bound
    

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
        n = np.arange(config.N)+1
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        delta = config.mu_max - suboptimal_mu
        delta_min = np.min(delta)
        
        if type(regret) == SamplingRegret:            
            bound = (48*config.K/delta_min)*np.log(np.true_divide(config.K*(n**4), self.delta)) + (10/3)*self.delta*config.K*np.log(np.true_divide(2*n, config.K))
            
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        if type(regret) == SwitchingRegret:
            bound = config.K*np.log(np.true_divide(2*n, config.K))
            
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        if type(regret) == TotalRegret:
            bound = np.true_divide(48*config.K*np.log(np.true_divide(config.K*(n**4), self.delta)), delta_min) + (10/3)*self.delta*config.K*np.log(np.true_divide(2*n, config.K)) + config.K*np.log(np.true_divide(2*n, config.K))
            
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        print("Error! Default implementation executed!")
        
    

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
        
    
    #constants are missing
    def theoretical_bound(self, config, regret=None):
        n = np.arange(config.N)+2
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        delta = config.mu_max - suboptimal_mu
        kl_vect = np.vectorize(self.kl)
        
        if type(regret) == SamplingRegret:   
            bound = (np.log(n)+np.log(np.log(n))) * np.sum(np.true_divide(delta, kl_vect(suboptimal_mu, config.mu_max)))
            return bound
        
        if type(regret) == SwitchingRegret:
            bound = 2*(np.log(n)+np.log(np.log(n))) * np.sum(np.true_divide(1, kl_vect(suboptimal_mu, config.mu_max)))
            return bound
        
        if type(regret) == TotalRegret:
            bound = (np.log(n)+np.log(np.log(n))) * np.sum(np.true_divide(delta, kl_vect(suboptimal_mu, config.mu_max))) + 2*(np.log(n)+np.log(np.log(n))) * np.sum(np.true_divide(1, kl_vect(suboptimal_mu, config.mu_max)))
            return bound
        
        print("Error! Default implementation executed!")

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