# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:17:58 2018

@author: lucas
"""
import policies as p
import regrets as r
import configurations as c

# log file
LOG_FILENAME = 'log_info.log'

# experiments setting
N = 2**17
num_repetitions=20
configs = [
            c.MUUD_Configuration(N),
#            c.MUUS_Configuration(N),
#            c.MCUD_Configuration(N)
            ]
policies = [
#            p.UCB1_Policy(), 
            p.UCB2_Policy(),
#            p.TS_Policy(),
#            p.TS2_Policy(),
#            p.OptTS_Policy(),
#            p.OptTS2_Policy(),
#            p.KLUCB_Policy(),
#            p.KLUCB2_Policy(),
#            p.UCYCLE_Policy(),
#            p.MOSS_Policy(),
#            p.UCBV_Policy(),
#            p.BayesUCB_Policy(),
#            p.BayesUCB2_Policy(),
#            p.UCB1SC_Policy(),
#            p.TSSC_Policy(),
#            p.BayesUCBSC_Policy(),
#            p.BayesUCB2SC_Policy(),
#            p.TS2SC_Policy(),
            p.UCB2SC_Policy()
            ]

# plot setting
budget = 2**4
regrets = [
            r.SamplingRegret(),
            r.SwitchingRegret(),
            r.TotalRegret(),
#            r.BudgetRegret(budget),
#            r.BudgetedReward(budget)
            ]
plot_th_bounds = False
sample_rate = 2**7