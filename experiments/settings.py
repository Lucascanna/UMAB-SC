# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:17:58 2018

@author: lucas
"""

from policies import UCB1_Policy, UCB2_Policy, TS_Policy, TS2_Policy, KLUCB_Policy, UCYCLE_Policy, UCB1SC_Policy, MOSS_Policy, UCBV_Policy, BayesUCB_Policy, OptTS_Policy, KLUCB2_Policy, BayesUCB2_Policy, OptTS2_Policy
from regrets import SamplingRegret, SwitchingRegret, TotalRegret, BudgetRegret
from configurations import MUUD_Configuration, MUUS_Configuration

# log file
LOG_FILENAME = 'log_info.log'

# experiments setting
N = 2**17
num_repetitions=20
budget = False
configs = [
            MUUD_Configuration(N),
#            MUUS_Configuration(N)
            ]
policies = [
            UCB1_Policy(), 
            UCB2_Policy(),
            TS_Policy(),
            TS2_Policy(),
            OptTS_Policy(),
            OptTS2_Policy(),
            KLUCB_Policy(),
            KLUCB2_Policy(),
            UCYCLE_Policy(),
            MOSS_Policy(),
            UCBV_Policy(),
            BayesUCB_Policy(),
            BayesUCB2_Policy(),
            UCB1SC_Policy()
            ]

# plot setting
budget = 2**9
regrets = [
            SamplingRegret(),
            SwitchingRegret(),
#            TotalRegret(),
            BudgetRegret(budget)
            ]
plot_th_bounds = False
sample_rate = 2**7