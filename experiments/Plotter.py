# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:33:32 2018

@author: lucas
"""

import matplotlib.pyplot as plt

class Plotter():
    """
    The class in charge of plotting the resulting regrets
    """
    
    def plot_regret(self, regrets, n, error, colors, err_colors, th_bounds, titles, labels, filename):
        """
        Each subplot refers to a specific regret.
        In each subplot the regret of all the policies
        """
        
        num_lines = regrets.shape[0]
        num_plots = regrets.shape[1]
    
        plt.figure(1, figsize=(10.5, 1.7*num_plots))
            
        for ii in range(num_plots):
            plt.subplot(num_plots,1,ii+1)
            plt.title(titles[ii])
            plt.xlabel('t')
            lines=[]
            for jj in range(num_lines):
                line = plt.errorbar(n, regrets[jj, ii, :], yerr=error[jj, ii, :], color=colors[jj], linestyle='-', ecolor=err_colors[jj], label=labels[jj])[0]
                if (th_bounds is not None):
                    plt.plot(n, th_bounds[jj, ii, :], color=colors[jj], linestyle='--')
                lines.append(line)
                
        plt.subplots_adjust(top=2, hspace=0.5)
        plt.figlegend(labels=labels, handles=lines, loc='upper right')
        
        plt.savefig("img/"+ filename +".png", format='png', bbox_inches='tight')
        plt.show()
            