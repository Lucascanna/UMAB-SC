# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:33:32 2018

@author: lucas
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors

class Plotter():
    """
    The class in charge of plotting the resulting regrets
    """
    
    def plot_regret(self, regrets, titles, policies, filename):
        """
        Each subplot refers to a specific regret.
        In each subplot the regret of all the policies
        """
        
        num_lines = regrets.shape[0]
        num_plots = regrets.shape[1]
        N = regrets.shape[2]
        
        n = np.arange(N)
        
        colors = list(mcolors.BASE_COLORS.keys())
        labels = []
        for policy in policies:
            labels = labels + [policy.name]
    
        plt.figure(1, figsize=(3.5*num_plots,5))
            
        for ii in range(num_plots):
            plt.subplot(num_plots,1,ii+1)
            plt.title(titles[ii])
            plt.xlabel('t')
            lines=[]
            for jj in range(num_lines):
                line = plt.plot(n, regrets[jj, ii, :], colors[jj]+'-', label=labels[jj])[0]
                lines.append(line)
                
        plt.subplots_adjust(top=2, hspace=0.5)
        plt.figlegend(labels=labels, handles=lines, loc='upper right')
        
        plt.savefig("img/"+ filename +".png", format='png', bbox_inches='tight')
        plt.show()
            