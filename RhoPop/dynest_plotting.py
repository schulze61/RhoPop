# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:04:39 2023

@author: jgsch
"""

from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
import functions as f

plot_path = '../output_plots/'


class continuum:
    
    def runplot(results, output_files_root = 'continuum'):
            fig, axes = dyplot.runplot(results) 
            plt.tight_layout()
            plt.savefig(plot_path + output_files_root + '_runplot' + '.png')
            

    def corner(results, output_files_root = 'continuum'):
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.reshape((2, 2))  # reshape axes
        
            # add white space
            for i in range(0, np.shape(axes)[0]):
                for k in range(0, np.shape(axes)[1]):
                	axes[i,k].tick_params(which = 'both', top = True, right = True, length = 10, width = 1, labelsize = 16, direction = 'in')
                	axes[i,k].patch.set_edgecolor('black')  
                	axes[i,k].patch.set_linewidth('3')
        
        
            fig, ax = dyplot.cornerplot(results, color='blue', dims = [0,1],
        				truth_color='black', show_titles=True, max_n_ticks=5, quantiles=None,
        				fig=(fig, axes[:, :2]), label_kwargs = {'fontsize': 16}, title_kwargs = {'fontsize': 16}, 
        				labels = [r'$\mu_{comp}$', r'$\sigma \rho_{ratio,pop}$'])
        
            axes[1,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[1,0].set_xticklabels([str(round(f.grids.cmf_to_wat(0.0), 2)), 
        			str(round(f.grids.cmf_to_wat(0.145), 2)), '0.3', '0.5', '0.75', '1.0'], fontsize = 16)
            axes[0,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
        
            plt.tight_layout()   
            plt.savefig(plot_path + output_files_root + '_hyperparam_corner' + '.png') 
            
            
    def traceplot(results, output_files_root = 'continuum'):           
        
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.reshape((2, 2))  # reshape axes
        
            fig, axes = dyplot.traceplot(results, dims = [0,1], truth_color='black', show_titles=True,
        			trace_cmap='viridis', connect=True, label_kwargs = {'fontsize': 18}, title_kwargs = {'fontsize': 18}, 
        			labels = [r'$\mu_{comp}$', r'$\sigma \rho_{ratio,pop}$'], 
        			connect_highlight=range(5), fig=(fig, axes))  
        
        
            plt.tight_layout()
            plt.savefig(plot_path + output_files_root + '_traceplot' + '.png')
