# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:58:28 2018

@author: lucas
"""

class UMUUD_Configuration(MUUD_Configuration):
    
    def __init__(self, N):
        super().__init__(N)
        self.name = 'MUUD'
        self.description = 'MUUD: Many arms, Uniform true mean rewards, Uniform switching costs, Deterministic switching costs'
        self.gen_unimodal_graph()

    
    def gen_unimodal_graph(self):  
        
        max_degree = 5
        
        self.neighborhood = dict.fromkeys(range(self.K), np.array([]))
        subopt_sorted = self.mu.argsort()[::-1][1:]
        better_candidates = np.array([self.best_arm])
        
        # build the unimodal paths
        for arm in subopt_sorted:
            better_candidates = self.update_candidates(better_candidates, max_degree)
            better = np.random.choice(better_candidates, 1, replace=False)
            self.connect_arms(arm, better)
            better_candidates = np.append(better_candidates, arm)
        
        random_candidates = self.update_candidates(np.arange(self.K), max_degree)
        # add random edges
        for arm in random_candidates:
            if arm in random_candidates:
                arm_candidates = np.delete(random_candidates, np.where(random_candidates == arm))
                num_random = np.random.randint(0, max_degree - self.neighborhood[arm].size + 1)
                random = np.random.choice(arm_candidates, num_random, replace=False)
                self.connect_arms(arm, random)
                random_candidates = self.update_candidates(random_candidates, max_degree)
        
        #add self loops and adjust switching costs       
        for arm in range(self.K):
            self.neighborhood[arm] = np.append(self.neighborhood[arm], arm)
            for to_arm in range(self.K):
                if not to_arm in self.neighborhood[arm]:
                    self.gamma[arm, to_arm] = float('+inf')
        self.switch_costs = self.gamma
            
            
        print(self.switch_costs)
    
    
    def connect_arms(self, from_arm, to_arms):
        self.neighborhood[from_arm] = np.unique(np.concatenate((self.neighborhood[from_arm], to_arms)))
        for arm in to_arms:
            if not from_arm in self.neighborhood[arm]:
                self.neighborhood[arm] = np.append(self.neighborhood[arm], from_arm)
                
    def update_candidates(self, candidates, max_degree):
        result = candidates
        for cand in candidates:
            if self.neighborhood[cand].size == max_degree:
                result = np.delete(result, np.where(result == cand))
        return result