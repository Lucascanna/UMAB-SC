# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:02:18 2018

@author: lucas
"""

from abc import ABC, abstractmethod
from math import e, log, ceil, pi
from scipy.special import btdtri
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
    
    def reset_state(self, K, N):
        """
        Resets the state of the policy for a new iteration
        """
        self.K = K
        self.N = N
    
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
    
    
    def tau(self, r):
        """
        Auxiliary function to compute epoch lengths
        """
        return np.ceil(np.power(1+self.alpha, r))
    
    
    

#%%
######### FREQUENTIST POLICIES ######################
#%%
        
class UCB1_Policy(Policy):
    """
    It chooses the next arm to play, according to UCB1 policy
    """
    
    def __init__(self):
        super().__init__()
        self.name='UCB1'
        self.color = 'gold'
        self.err_color = 'goldenrod'
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
#        self.indexes= np.zeros((N, K))
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            return n
        
        # loop phase
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        b = (np.true_divide(2*log(n), T))**0.5
        u = mu_hat + b
        
#        self.indexes[n,:] = u
        
        return np.argmax(u)
    
    
    
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
        
        raise TypeError("Regret type does not exists!")
        
    

        

#%%
    
class KLUCB_Policy(Policy):
    """
    It chooses the next arm to play, according to KL-UCB policy
    """
    
    def __init__(self, precision=1e-6, max_iterations=50):
        super().__init__()
        self.name='KLUCB_'+ str(precision) + '_' + str(max_iterations)
        self.color = 'b'
        self.err_color = 'lightblue'
        self.precision=precision
        self.max_iterations=max_iterations
        self.compute_indexes = np.vectorize(self.compute_arm_index)
    
    def reset_state(self, K, N):
        super().reset_state(K,N)
#        self.indexes= np.zeros((N, K))
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        # init phase
        if n < self.K:
            return n
        
        # loop phase
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        
        q = self.compute_indexes(mu_hat, np.log(n)/T)
#        self.indexes[n,:] = q
        
        return np.argmax(q)
    
    def compute_arm_index(self, x, d, upperbound=1, lowerbound=0):
        
        value = max(x, lowerbound)
        u = upperbound
        
        _count_iteration = 0
        while _count_iteration < self.max_iterations and u - value > self.precision:
            _count_iteration += 1
            m = (value + u) / 2.
            if self.kl(x, m) > d:
                u = m
            else:
                value = m
                
        return (value + u) / 2
    
    
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
        
        raise TypeError("Regret type does not exists!")
        
#%%

class MOSS_Policy(Policy):
    """
    It chooses the next arm to play, according to MOSS policy.
    """
    
    def __init__(self):
        super().__init__()
        self.name='MOSS'
        self.color = 'g'
        self.err_color = 'lightgreen'
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            return n
        
        # loop phase
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        b = (np.vectorize(max)(np.log(n/(self.K*T)), 0) / T)**0.5
        u = mu_hat + b
        
        return np.argmax(u)
    
    
    def theoretical_bound(self, config, regret=None):
        n = np.arange(config.N)+1
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        delta = config.mu_max - suboptimal_mu
        delta_min = np.min(delta)
        
        if type(regret) == SamplingRegret:            
            bound = ((23*config.K)/delta_min)*np.log(np.vectorize(max)((110*(delta_min**2)*n)/config.K, 10**4))
            
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        if type(regret) == SwitchingRegret:
            bound = ((46*config.K)/(delta_min**2))*np.log(np.vectorize(max)((110*(delta_min**2)*n)/config.K, 10**4))
            
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        if type(regret) == TotalRegret:
            bound = ((23*config.K)/delta_min)*np.log(np.vectorize(max)((110*(delta_min**2)*n)/config.K, 10**4)) + ((46*config.K)/(delta_min**2))*np.log(np.vectorize(max)((110*(delta_min**2)*n)/config.K, 10**4))
            
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        raise TypeError("Regret type does not exist!")
        
#%%

class UCBV_Policy(Policy):
    """
    It chooses the next arm to play, according to UCBV policy
    Explration function: log(n)
    c = 1
    """
    
    def __init__(self):
        super().__init__()
        self.name='UCBV'
        self.color = 'm'
        self.err_color = 'plum'
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            return n
        
        # loop phase
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        dev_hat = p0*mu_hat**2 + p1*(1-mu_hat)**2
        var_hat = np.true_divide(dev_hat, T)
        b = (np.true_divide(2*var_hat*log(n), T))**0.5 + (3*log(n))/T
        u = mu_hat + b
        
        
        return np.argmax(u)
    
    
    
    def theoretical_bound(self, config, regret=None):
        n = np.arange(config.N)+config.K
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        delta = config.mu_max - suboptimal_mu
        suboptimal_var = suboptimal_mu * (1-suboptimal_mu)
        one_on_t = np.sum(1/n)
        
        if type(regret) == SamplingRegret:  
            bound = 0
            for ii in range(config.K - 1):
                bound += (1 + 8*(suboptimal_var[ii]/delta[ii]**2 + 2/delta[ii])*np.log(n) + 24*suboptimal_var[ii]/delta[ii]**2 + 4/delta[ii] + one_on_t)*delta[ii]
            bound = np.insert(bound[:-config.K], 0, arms)
            return bound
        
        if type(regret) == SwitchingRegret:
            bound = 16*np.log(n)*np.sum(np.true_divide(1, delta**2))
            
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        if type(regret) == TotalRegret:
            bound_sam = 0
            for ii in range(config.K - 1):
                bound_sam += (1 + 8*(suboptimal_var[ii]/delta[ii]**2 + 2/delta[ii])*np.log(n) + 24*suboptimal_var[ii]/delta[ii]**2 + 4/delta[ii] + one_on_t)*delta[ii]
            bound_sam = np.insert(bound_sam[:-config.K], 0, arms)
            bound_sw = 16*np.log(n)*np.sum(np.true_divide(1, delta**2))
            bound_sw = np.insert(bound_sw[:-1],0,0)
            return bound_sam + bound_sw
        
        raise TypeError("Regret type does not exist!")

#%%
        
############### BAYESIAN POLICIES ##############################

#%%
    
class TS_Policy(Policy):
    """
    It chooses the next arm to play, according to Thompson Sampling policy
    """
    
    def __init__(self):
        super().__init__()
        self.name='TS'
        self.color = 'r'
        self.err_color = 'tomato'
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        samples = np.random.beta(p1 + 1, p0 + 1)
        return np.argmax(samples)
        
    
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
        
        raise TypeError("Regret type does not exist!")

    
#%%

class OptTS_Policy(Policy):
    """
    It chooses the next arm to play, according to Optimistic Thompson Sampling policy
    """
    
    def __init__(self):
        super().__init__()
        self.name='OptTS'
        self.color = 'darkorange'
        self.err_color = 'orange'
        self.max_vect = np.vectorize(max)
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            return n
        
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        
        samples = self.max_vect(np.random.beta(p1 + 1, p0 + 1), mu_hat)
        return np.argmax(samples)
        
    
    #constants are missing
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
        

#%%
    
class BayesUCB_Policy(Policy):
    """
    It chooses the next arm to play, according to Bayes-UCB policy
    """
    
    def __init__(self):
        super().__init__()
        self.name='BayesUCB'
        self.color = 'indigo'
        self.err_color = 'mediumpurple'
        self.quantile = np.vectorize(btdtri)
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
#        self.indexes= np.zeros((N, K))
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        quantiles = self.quantile(p1 + 1, p0 + 1, 1-(1/max(1,n)))
        
#        self.indexes[n,:] = quantiles

        return np.argmax(quantiles)
        
    
    def theoretical_bound(self, config, regret=None):
        n = np.arange(config.N)+1
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        
        delta = config.mu_max - suboptimal_mu
        kl_vect = np.vectorize(self.kl)
        d = kl_vect(suboptimal_mu, config.mu_max)
        
        if type(regret) == SamplingRegret:   
            bound = np.log(n)*np.sum(np.true_divide(delta, d))
            bound = np.insert(bound[:-1],0,0)
            return bound
        
        if type(regret) == SwitchingRegret:
            bound = n
            return bound
        
        if type(regret) == TotalRegret:
            bound = n
            return bound
        
        raise TypeError("Regret type does not exist!")
        
#%%
        
#################### EPOCH-STYLE POLICIES ############################
        
#%%
        
class UCB2_Policy(Policy):
    """
    It chooses the next arm to play, according to UCB2 policy
    """
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.name='UCB2_'+ str(alpha)
        self.color = 'k'
        self.err_color = 'grey'
        self.alpha = alpha
        
        
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.ones(K, dtype=np.int32)
#        self.indexes= np.zeros((N, K))
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            self.current_arm=n
            return n
        
        # loop phase
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
#            self.indexes[n,:] = self.indexes[n-1,:]
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:
            
            #choose arm    
            T = p0 + p1
            mu_hat = np.true_divide(p1, T)
            b = np.true_divide((1+self.alpha) * np.log(np.true_divide((e*n), self.tau(self.epochs_arm))),
                                2*self.tau(self.epochs_arm)) ** 0.5
            u = mu_hat + b
            
#            self.indexes[n,:] = u
            
            #update state
            self.current_arm = np.argmax(u)
            self.epoch_steps = 1
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")

    
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
        
        raise TypeError("Regret type does not exists!")
        
        
#%%
    
class UCYCLE_Policy(Policy):
    """
    It chooses the next arm to play, according to UCYCLE policy
    """
    
    def __init__(self, delta=0.004):
        super().__init__()
        self.name='UCYCLE'
        self.color = 'c'
        self.err_color = 'lavender'
        self.delta= delta
        
    
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        
    
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
            #b = np.true_divide(np.log(np.true_divide(self.K * n**4, self.delta)),
             #                   2*T) ** 0.5
            b = np.true_divide(log(self.K * n), 2*T) ** 0.5 #delta= n^3
            u = mu_hat + b
            
            #update state
            self.current_arm = np.argmax(u)
            self.epoch_steps = 1
            self.epoch_len = T[self.current_arm]
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
    
    
    def theoretical_bound(self, config, regret=None):
        n = np.arange(config.N)+1
        arms = np.arange(config.K)
        suboptimal_arms = np.delete(arms, config.best_arm)
        suboptimal_mu = config.mu[suboptimal_arms]
        delta = config.mu_max - suboptimal_mu
        delta_min = np.min(delta)
        
        if type(regret) == SamplingRegret:            
            bound = (48*config.K/delta_min)*np.log(config.K*n) + (10/3)*n**3*config.K*np.log(np.true_divide(2*n, config.K))
            
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
        
        raise TypeError("Regret type does not exist!")
        

#%%    

################## NEW EPOCH-STYLE POLICIES ##############################

#%%


class TS2_Policy(Policy):
    """
    It chooses the next arm to play, according to Thompson Sampling policy in epochs
    """
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.name='TS2_' + str(alpha)
        self.color = 'pink'
        self.err_color = 'lightpink'
        self.alpha = alpha
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.zeros(K, dtype=np.int32)
        
    
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
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
    
    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
    
#%%
        
class KLUCB2_Policy(Policy):
    """
    It chooses the next arm to play, according to KL-UCB2 policy
    """
    
    def __init__(self, precision=1e-6, max_iterations=50, alpha=0.1):
        super().__init__()
        self.name='KLUCB2_'+ str(precision) + '_' + str(max_iterations) + '_' + str(alpha)
        self.color = 'sienna'
        self.err_color = 'peru'
        self.compute_indexes = np.vectorize(self.compute_arm_index)
        self.precision=precision
        self.max_iterations=max_iterations
        self.alpha = alpha
        
    
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.ones(K, dtype=np.int32)
        
    
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
            
            #choose arm    
            T = p0 + p1
            mu_hat = np.true_divide(p1, T)

            q = self.compute_indexes(mu_hat, np.log(n)/T)
            
            #update state
            self.current_arm = np.argmax(q)
            self.epoch_steps = 1
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
        
    
    def compute_arm_index(self, x, d, upperbound=1, lowerbound=0):
        
        value = max(x, lowerbound)
        u = upperbound
        
        _count_iteration = 0
        while _count_iteration < self.max_iterations and u - value > self.precision:
            _count_iteration += 1
            m = (value + u) / 2.
            if self.kl(x, m) > d:
                u = m
            else:
                value = m
                
        return (value + u) / 2
    
    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
    
#%%

class BayesUCB2_Policy(Policy):
    """
    It chooses the next arm to play, according to Bayes-UCB2 policy
    """
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.name='BayesUCB2_' + str(alpha)
        self.color = 'orangered'
        self.err_color = 'coral'
        self.quantile = np.vectorize(btdtri)
        self.alpha = alpha
        
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.zeros(K, dtype=np.int32)
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            self.current_arm=n
            return n
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:
            
            #choose arm    
            quantiles = self.quantile(p1 + 1, p0 + 1, 1-(1/n))
            
            #update state
            self.current_arm = np.argmax(quantiles)
            self.epoch_steps = 1
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
        
    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
    
#%%

class OptTS2_Policy(Policy):
    """
    It chooses the next arm to play, according to Optimistic Thompson Sampling 2 policy
    """
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.name='OptTS2_' + str(alpha)
        self.color = 'olive'
        self.err_color = 'y'
        self.max_vect = np.vectorize(max)
        self.alpha=alpha
        
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.ones(K, dtype=np.int32)
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            self.current_arm=n
            return n
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:
            
            #choose arm    
            T = p0 + p1
            mu_hat = np.true_divide(p1, T)
        
            samples = self.max_vect(np.random.beta(p1 + 1, p0 + 1), mu_hat)
            
            #update state
            self.current_arm = np.argmax(samples)
            self.epoch_steps = 1
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
        
    
    #constants are missing
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
        
#%%
        
################# NEW SWITCHING COST POLICIES #############################
        
#%%
        
class UCB1SC_Policy(Policy):
    """
    It chooses the next arm to play, according to UCB1 policy.
    The index also takes into account cost for switching
    """
    
    def __init__(self):
        super().__init__()
        self.name='UCB1SC'
        self.color = 'darkgrey'
        self.err_color = 'lightgrey'
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.init_epoch = 0
        self.init_time = 0
        self.end_init_phase = K**2 - 1
        self.current_arm = -1
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        
        # init phase: too many switches
#        if n < self.end_init_phase:  
#            
#            if self.init_time > 2*(self.K - self.init_epoch - 1):
#                self.init_epoch += 1
#                self.init_time = 1
#                self.current_arm = self.init_epoch
#            else:
#                if self.init_time % 2 == 0:
#                    self.current_arm = self.init_epoch
#                else:
#                    self.current_arm = ceil(self.init_time/2) + self.init_epoch
#                
#                self.init_time += 1
#            return self.current_arm
        
        #init phase
        if n < self.K:
            self.current_arm=n
            return n
        
        # loop phase
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        b = (np.true_divide(2*log(n), T))**0.5
        if s0 is None:
            gamma_hat = s1[self.current_arm,:]
            c = gamma_hat/(self.N - n)
        else:
            curr_s0 = s0[self.current_arm, :]
            curr_s1 = s1[self.current_arm,:]
            curr_tot = curr_s0 + curr_s1
            gamma_hat = np.ones_like(curr_tot, dtype=np.float)
            gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
            b_cost = np.zeros_like(gamma_hat)
            b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
            c = (gamma_hat - b_cost)/(self.N - n)
        u = mu_hat + b - c
        
        self.current_arm = np.argmax(u)
        return self.current_arm

    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
        
#%%
    
class KLUCBSC_Policy(Policy):
    """
    It chooses the next arm to play, according to KL-UCB policy.
    It also considers switching costs
    """
    
    def __init__(self, precision=1e-6, max_iterations=50):
        super().__init__()
        self.name='KLUCBSC_'+ str(precision) + '_' + str(max_iterations)
        self.color = 'navy'
        self.err_color = 'cornflowerblue'
        self.precision=precision
        self.max_iterations=max_iterations
        self.compute_indexes = np.vectorize(self.compute_arm_index)
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        # init phase
        if n < self.K:
            return n
        
        # loop phase
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        
        q = self.compute_indexes(mu_hat, np.log(n)/T)
        if s0 is None:
            gamma_hat = s1[self.current_arm,:]
            c = gamma_hat/(self.N - n)
        else:
            curr_s0 = s0[self.current_arm, :]
            curr_s1 = s1[self.current_arm,:]
            curr_tot = curr_s0 + curr_s1
            gamma_hat = np.ones_like(curr_tot, dtype=np.float)
            gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
            b_cost = np.zeros_like(gamma_hat)
            b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
            c = (gamma_hat - b_cost)/(self.N - n)
            
        b = q - c
        return np.argmax(b)
    
    def compute_arm_index(self, x, d, upperbound=1, lowerbound=0):
        
        value = max(x, lowerbound)
        u = upperbound
        
        _count_iteration = 0
        while _count_iteration < self.max_iterations and u - value > self.precision:
            _count_iteration += 1
            m = (value + u) / 2.
            if self.kl(x, m) > d:
                u = m
            else:
                value = m
                
        return (value + u) / 2
    
    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError


#%%

class TSSC_Policy(Policy):
    """
    It chooses the next arm to play, according to Thompson Sampling policy.
    It also conseders switching costs
    """
    
    def __init__(self):
        super().__init__()
        self.name='TSSC'
        self.color = 'indianred'
        self.err_color = 'lightcoral'
    
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.init_epoch = 0
        self.init_time = 0
        self.end_init_phase = K**2 - 1
        self.current_arm = -1
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        #loop phase
        samples = np.random.beta(p1 + 1, p0 + 1)
        if s0 is None:
            gamma_hat = s1[self.current_arm,:]
            c = gamma_hat/(self.N - n)
        else:
            curr_s0 = s0[self.current_arm, :]
            curr_s1 = s1[self.current_arm,:]
            curr_tot = curr_s0 + curr_s1
            gamma_hat = np.ones_like(curr_tot, dtype=np.float)
            gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
            b_cost = np.zeros_like(gamma_hat)
            b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
            c = (gamma_hat - b_cost)/(self.N - n)
            
        b = samples - c
        self.current_arm = np.argmax(b)
        return self.current_arm
        
    
    #constants are missing
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError

#%%

class OptTSSC_Policy(Policy):
    """
    It chooses the next arm to play, according to Optimistic Thompson Sampling policy.
    It also considers switching costs.
    """
    
    def __init__(self):
        super().__init__()
        self.name='OptTSSC'
        self.color = 'maroon'
        self.err_color = 'salmon'
        self.max_vect = np.vectorize(max)
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            return n
        
        T = p0 + p1
        mu_hat = np.true_divide(p1, T)
        
        samples = self.max_vect(np.random.beta(p1 + 1, p0 + 1), mu_hat)
        if s0 is None:
            gamma_hat = s1[self.current_arm,:]
            c = gamma_hat/(self.N - n)
        else:
            curr_s0 = s0[self.current_arm, :]
            curr_s1 = s1[self.current_arm,:]
            curr_tot = curr_s0 + curr_s1
            gamma_hat = np.ones_like(curr_tot, dtype=np.float)
            gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
            b_cost = np.zeros_like(gamma_hat)
            b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
            c = (gamma_hat - b_cost)/(self.N - n)
        b = samples - c
        return np.argmax(b)
        
    
    #constants are missing
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError


#%%
        
class BayesUCBSC_Policy(Policy):
    """
    It chooses the next arm to play, according to Bayes-UCB policy.
    It also considers switching costs.
    """
    
    def __init__(self):
        super().__init__()
        self.name='BayesUCBSC'
        self.color = 'mediumorchid'
        self.err_color = 'thistle'
        self.quantile = np.vectorize(btdtri)
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.init_epoch = 0
        self.init_time = 0
        self.end_init_phase = K**2 - 1
        self.current_arm = -1


        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        quantiles = self.quantile(p1 + 1, p0 + 1, 1-(1/max(1,n)))
        if s0 is None:
            gamma_hat = s1[self.current_arm,:]
            c = gamma_hat/(self.N - n)
        else:
            curr_s0 = s0[self.current_arm, :]
            curr_s1 = s1[self.current_arm,:]
            curr_tot = curr_s0 + curr_s1
            gamma_hat = np.ones_like(curr_tot, dtype=np.float)
            gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
            b_cost = np.zeros_like(gamma_hat)
            b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
            c = (gamma_hat - b_cost)/(self.N - n)
        b= quantiles - c
        self.current_arm = np.argmax(b)
        return self.current_arm
        
    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
        

        
#%%

class UCB2SC_Policy(Policy):
    """
    It chooses the next arm to play, according to UCB2 policy.
    It also considers switching costs
    """
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.name='UCB2SC_'+ str(alpha)
        self.color = 'dimgray'
        self.err_color = 'darkgray'
        self.alpha = alpha
        
        
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.ones(K, dtype=np.int32)
        
    
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
            
            #choose arm    
            T = p0 + p1
            mu_hat = np.true_divide(p1, T)
            b = np.true_divide((1+self.alpha) * np.log(np.true_divide((e*n), self.tau(self.epochs_arm))),
                                2*self.tau(self.epochs_arm)) ** 0.5
            if s0 is None:
                gamma_hat = s1[self.current_arm,:]
                c = gamma_hat/(self.N - n)
            else:
                curr_s0 = s0[self.current_arm, :]
                curr_s1 = s1[self.current_arm,:]
                curr_tot = curr_s0 + curr_s1
                gamma_hat = np.ones_like(curr_tot)
                gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
                b_cost = np.zeros_like(gamma_hat)
                b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
                c = (gamma_hat - b_cost)/(self.N - n)
            u = mu_hat + b - c
            
            #update state
            self.current_arm = np.argmax(u)
            self.epoch_steps = 1
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")

    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
        
#%%
    
class UCYCLESC_Policy(Policy):
    """
    It chooses the next arm to play, according to UCYCLE policy.
    It also considers switchin costs.
    """
    
    def __init__(self, delta=0.004):
        super().__init__()
        self.name='UCYCLESC'
        self.color = 'cadetblue'
        self.err_color = 'powderblue'
        self.delta= delta
        
    
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        
    
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
            #b = np.true_divide(np.log(np.true_divide(self.K * n**4, self.delta)),
             #                   2*T) ** 0.5
            b = np.true_divide(log(self.K * n), 2*T) ** 0.5 #delta= n^3
            if s0 is None:
                gamma_hat = s1[self.current_arm,:]
                c = gamma_hat/(self.N - n)
            else:
                curr_s0 = s0[self.current_arm, :]
                curr_s1 = s1[self.current_arm,:]
                curr_tot = curr_s0 + curr_s1
                gamma_hat = np.ones_like(curr_tot)
                gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
                b_cost = np.zeros_like(gamma_hat)
                b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
                c = (gamma_hat - b_cost)/(self.N - n)
            u = mu_hat + b - c
            
            #update state
            self.current_arm = np.argmax(u)
            self.epoch_steps = 1
            self.epoch_len = T[self.current_arm]
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
    
    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError

#%%
        
################ NEW EPOCH-STYLE WITH SWITCHING COSTS POLICIES ##############

#%%
class KLUCB2SC_Policy(Policy):
    """
    It chooses the next arm to play, according to epoch-style KL-UCB policy.
    It also considers switching costs.
    """
    
    def __init__(self, precision=1e-6, max_iterations=50, alpha=0.1):
        super().__init__()
        self.name='KLUCB2SC_'+ str(precision) + '_' + str(max_iterations) + '_' + str(alpha)
        self.color = 'sandybrown'
        self.err_color = 'mocassin'
        self.compute_indexes = np.vectorize(self.compute_arm_index)
        self.precision=precision
        self.max_iterations=max_iterations
        self.alpha = alpha
        
    
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.ones(K, dtype=np.int32)
        
    
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
            
            #choose arm    
            T = p0 + p1
            mu_hat = np.true_divide(p1, T)

            q = self.compute_indexes(mu_hat, np.log(n)/T)
            if s0 is None:
                gamma_hat = s1[self.current_arm,:]
                c = gamma_hat/(self.N - n)
            else:
                curr_s0 = s0[self.current_arm, :]
                curr_s1 = s1[self.current_arm,:]
                curr_tot = curr_s0 + curr_s1
                gamma_hat = np.ones_like(curr_tot)
                gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
                b_cost = np.zeros_like(gamma_hat)
                b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
                c = (gamma_hat - b_cost)/(self.N - n)
            b = q - c
            
            #update state
            self.current_arm = np.argmax(b)
            self.epoch_steps = 1
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
        
    
    def compute_arm_index(self, x, d, upperbound=1, lowerbound=0):
        
        value = max(x, lowerbound)
        u = upperbound
        
        _count_iteration = 0
        while _count_iteration < self.max_iterations and u - value > self.precision:
            _count_iteration += 1
            m = (value + u) / 2.
            if self.kl(x, m) > d:
                u = m
            else:
                value = m
                
        return (value + u) / 2
    
    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
#%%

class BayesUCB2SC_Policy(Policy):
    """
    It chooses the next arm to play, according to Bayes-UCB in epochs.
    It also considers switching costs.
    """
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.name='BayesUCB2SC_' + str(alpha)
        self.color = 'darkslateblue'
        self.err_color = 'slateblue'
        self.quantile = np.vectorize(btdtri)
        self.alpha = alpha
        
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.zeros(K, dtype=np.int32)
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            self.current_arm=n
            return n
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:
            
            #choose arm
            quantiles = self.quantile(p1 + 1, p0 + 1, 1-(1/n))
            if s0 is None:
                gamma_hat = s1[self.current_arm,:]
                c = gamma_hat/(self.N - n)
            else:
                curr_s0 = s0[self.current_arm, :]
                curr_s1 = s1[self.current_arm,:]
                curr_tot = curr_s0 + curr_s1
                gamma_hat = np.ones_like(curr_tot)
                gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
                b_cost = np.zeros_like(gamma_hat)
                b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
                c = (gamma_hat - b_cost)/(self.N - n)
            b = quantiles - c
            
            #update state
            self.current_arm = np.argmax(b)
            self.epoch_steps = 1
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
        
    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
        
#%%

class TS2SC_Policy(Policy):
    """
    It chooses the next arm to play, according to Thompson Sampling policy in epochs.
    It also considers swithing costs.
    """
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.name='TS2SC_' + str(alpha)
        self.color = 'deeppink'
        self.err_color = 'violet'
        self.alpha = alpha
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.zeros(K, dtype=np.int32)
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:                
                
            samples = np.random.beta(p1 + 1, p0 + 1)
            if s0 is None:
                gamma_hat = s1[self.current_arm,:]
                c = gamma_hat/(self.N - n)
            else:
                curr_s0 = s0[self.current_arm, :]
                curr_s1 = s1[self.current_arm,:]
                curr_tot = curr_s0 + curr_s1
                gamma_hat = np.ones_like(curr_tot)
                gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
                b_cost = np.zeros_like(gamma_hat)
                b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
                c = (gamma_hat - b_cost)/(self.N - n)
            b = samples - c
            
            
            #update state
            self.current_arm = np.argmax(b)
            self.epoch_steps = 1
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
    
    
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
        
#%%

class OptTS2SC_Policy(Policy):
    """
    It chooses the next arm to play, according to Optimistic Thompson Sampling 2 policy.
    It also considers the switcing costs.
    """
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.name='OptTS2SC_' + str(alpha)
        self.color = 'yellow'
        self.err_color = 'khaki'
        self.max_vect = np.vectorize(max)
        self.alpha=alpha
        
        
    def reset_state(self, K, N):
        super().reset_state(K,N)
        self.current_arm = -1
        self.epoch_len = 0
        self.epoch_steps = 0
        self.epochs_arm = np.ones(K, dtype=np.int32)
        
    
    def choose_arm(self, n, p0, p1, s0, s1):
        
        # init phase
        if n < self.K:
            self.current_arm=n
            return n
        
        # epoch play
        if self.epoch_steps < self.epoch_len:
            self.epoch_steps += 1
            return self.current_arm
        
        # epoch change
        if self.epoch_steps == self.epoch_len:
            
            #choose arm    
            T = p0 + p1
            mu_hat = np.true_divide(p1, T)
        
            samples = self.max_vect(np.random.beta(p1 + 1, p0 + 1), mu_hat)
            if s0 is None:
                gamma_hat = s1[self.current_arm,:]
                c = gamma_hat/(self.N - n)
            else:
                curr_s0 = s0[self.current_arm, :]
                curr_s1 = s1[self.current_arm,:]
                curr_tot = curr_s0 + curr_s1
                gamma_hat = np.ones_like(curr_tot)
                gamma_hat = np.true_divide(curr_s1, curr_tot, where=(curr_tot!=0), out=gamma_hat)
                b_cost = np.zeros_like(gamma_hat)
                b_cost = np.true_divide(2*log(n), curr_tot, where=(curr_tot!=0), out=b_cost)**0.5
                c = (gamma_hat - b_cost)/(self.N - n)
                
            b= samples - c
            
            #update state
            self.current_arm = np.argmax(b)
            self.epoch_steps = 1
            self.epoch_len = max(1,self.tau(self.epochs_arm[self.current_arm]+1) - self.tau(self.epochs_arm[self.current_arm]))
            self.epochs_arm[self.current_arm] += 1
            
            return self.current_arm
        
        raise ValueError("Epoch steps greater than epoch length")
        
    
    #constants are missing
    def theoretical_bound(self, config, regret=None):
        raise NotImplementedError
        
    