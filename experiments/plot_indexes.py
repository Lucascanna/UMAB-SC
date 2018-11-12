# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:01:03 2018

@author: lucas
"""

import matplotlib.pyplot as plt
import pickle as pk
#%%

FILE_BAYES = 'results/20rep131072times/MUUD/BayesUCB/exp0.pkl'
FILE_UCB2 = 'results/20rep131072times/MUUD/UCB2_0.5/exp0.pkl'
FILE_UCB1 = 'results/20rep131072times/MUUD/UCB1/exp0.pkl'
FILE_KLUCB = 'results/20rep131072times/MUUD/KLUCB_1e-06_50/exp0.pkl'

bayes_exp = pk.load(open(FILE_BAYES, 'rb'))
ucb2_exp = pk.load(open(FILE_UCB2, 'rb'))
ucb1_exp = pk.load(open(FILE_UCB1, 'rb'))
klucb_exp = pk.load(open(FILE_KLUCB, 'rb'))

q = bayes_exp.policy.indexes
u2 = ucb2_exp.policy.indexes
u1 = ucb1_exp.policy.indexes
kl = klucb_exp.policy.indexes
x = range(2**17)
y = [u1, kl, q]

colors = ['gold', 'b', 'orangered']
labels = ['ucb1', 'klucb', 'bayesUCB']


#print(q[:,0])
#print(u[:,0])

num_plots = q.shape[1]
num_lines = len(y)
plt.figure(1, figsize=(10.5, 1.7*num_plots))
for ii in range(num_plots):
    plt.subplot(num_plots,1,ii+1)
    plt.title('ARM '+str(ii))
    plt.xlabel('t')
    plt.ylabel('ucb')
    plt.axis([0, 2**17, 0.6, 1.3])
    lines=[]
    for jj in range(num_lines):
        line, = plt.plot(x, y[jj][:,ii], colors[jj])
        lines.append(line)
        
plt.subplots_adjust(top=2, hspace=0.5)
plt.figlegend(labels=labels, handles=lines, loc='upper right')
plt.savefig("img/indexes2.png", format='png', bbox_inches='tight')

plt.show()