# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:04:39 2023
"""

from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
import functions as f

plot_path = './output_plots/'


class continuum:
    
    def runplot(results, froot = ''):
            fig, axes = dyplot.runplot(results) 
            plt.tight_layout()
            fn = plot_path + froot + '_continuum_runplot' + '.png'
            plt.savefig(fn)
            

    def corner(results, froot = ''):
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
            fn = plot_path + froot + '_continuum_hyperparam_corner' + '.png'
            plt.savefig(fn) 
            
            
    def traceplot(results, froot = ''):           
        
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.reshape((2, 2))  # reshape axes
        
            fig, axes = dyplot.traceplot(results, dims = [0,1], truth_color='black', show_titles=True,
        			trace_cmap='viridis', connect=True, label_kwargs = {'fontsize': 18}, title_kwargs = {'fontsize': 18}, 
        			labels = [r'$\mu_{comp}$', r'$\sigma \rho_{ratio,pop}$'], 
        			connect_highlight=range(5), fig=(fig, axes))  
        
        
            plt.tight_layout()
            fn = plot_path + froot + '_continuum_traceplot' + '.png'
            plt.savefig(fn)
            
            
class two_population:
    
    def runplot(results, froot = ''):    
        fig, axes = dyplot.runplot(results) 
        plt.tight_layout()
        fn = plot_path + froot + '_twopop_runplot' + '.png'
        plt.savefig(fn)


    def corner(results, froot = ''):
        fig, axes = plt.subplots(5, 5, figsize=(20, 20))
        axes = axes.reshape((5, 5))  # reshape axes
    
        # add white space
        for i in range(0, np.shape(axes)[0]):
            for k in range(0, np.shape(axes)[1]):
            	axes[i,k].tick_params(which = 'both', top = True, right = True, length = 10, width = 1, labelsize = 16, direction = 'in')
            	axes[i,k].patch.set_edgecolor('black')  
            	axes[i,k].patch.set_linewidth('3')
    
    
        fig, ax = dyplot.cornerplot(results, color='blue', dims = [0,1,2,3,4],
        truth_color='black', show_titles=True, max_n_ticks=5, quantiles=None,
    				fig=(fig, axes[:, :5]), label_kwargs = {'fontsize': 16}, title_kwargs = {'fontsize': 16}, 
    				labels = [r'$\mu_{comp}^{(1)}$', r'$\sigma \rho_{ratio,pop}^{(1)}$',  r'$\mu_{comp}^{(2)}$', r'$\sigma \rho_{ratio,pop}^{(2)}$', r'$\pi^{(1)}$'])
    
    
        #histogram of mu1
        axes[0,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
    
        #sigma mu1 vs mu1
        axes[1,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
    
        #mu2 vs mu1
        axes[2,0].set_yticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
        axes[2,0].set_yticklabels([str(round(f.grids.cmf_to_wat(0.0), 2)), 
    			str(round(f.grids.cmf_to_wat(0.145), 2)), '0.3', '0.5', '0.75', '1.0'], fontsize = 16)
        axes[2,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
    
        #mu2 vs sigma mu1
        axes[2,1].set_yticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
    
        #histogram of mu2
        axes[2,2].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
    
        #sigma mu2 vs mu1
        axes[3,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
    
        #sigma mu2 vs mu2
        axes[3,2].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])	
    
        #MW vs mu1
        axes[4,0].set_ylim(0,1)
        axes[4,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
        axes[4,0].set_xticklabels([str(round(f.grids.cmf_to_wat(0.0), 2)), 
    			str(round(f.grids.cmf_to_wat(0.145), 2)), '0.3', '0.5', '0.75', '1.0'], fontsize = 16)	
    
    
        #MW vs sigma mu1
        axes[4,1].set_ylim(0,1)
    
        #MW vs mu2
        axes[4,2].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
        axes[4,2].set_xticklabels([str(round(f.grids.cmf_to_wat(0.0), 2)), 
    			str(round(f.grids.cmf_to_wat(0.145), 2)), '0.3', '0.5', '0.75', '1.0'], fontsize = 16)	
        axes[4,2].set_ylim(0,1)	
    
        #MW vs sigma mu2
        axes[4,3].set_ylim(0,1)
    
    
        #Histogram of MW
        axes[4,4].set_xlim(0,1)			

 
        plt.tight_layout()
        fn = plot_path + froot + '_twopop_hyperparam_corner' + '.png'
        plt.savefig(fn)  
        

    
    def traceplot(results, froot = ''):  
        fig, axes = plt.subplots(5, 2, figsize=(10, 20))
        axes = axes.reshape((5, 2))  # reshape axes
    
    
        fig, axes = dyplot.traceplot(results, dims = [0,1, 2, 3, 4], truth_color='black', show_titles=True,
    			trace_cmap='viridis', connect=True, label_kwargs = {'fontsize': 18}, title_kwargs = {'fontsize': 18}, 
    			labels = [r'$\mu_{comp}^{(1)}$', r'$\sigma \rho_{ratio,pop}^{(1)}$',  r'$\mu_{comp}^{(2)}$', r'$\sigma \rho_{ratio,pop}^{(2)}$', r'$\pi^{(1)}$'], 
    			connect_highlight=range(5), fig=(fig, axes))  
    
    
        plt.tight_layout()
        fn = plot_path + froot + '_twopop_traceplot' + '.png'
        plt.savefig(fn)
        
        

class three_population:
    
    
    def runplot(results, froot = ''): 
        fig, axes = dyplot.runplot(results) 
        plt.tight_layout()
        fn = plot_path + froot + '_threepop_runplot' + '.png'
        plt.savefig(fn)

    def corner(results, froot = ''):
        fig, axes = plt.subplots(8, 8, figsize=(20, 20))
        axes = axes.reshape((8, 8))  # reshape axes
    
        # add white space
        for i in range(0, np.shape(axes)[0]):
            for k in range(0, np.shape(axes)[1]):
                axes[i,k].tick_params(which = 'both', top = True, right = True, length = 10, width = 1, labelsize = 16, direction = 'in')
                axes[i,k].patch.set_edgecolor('black')  
                axes[i,k].patch.set_linewidth('3')
    
    
            fig, ax = dyplot.cornerplot(results, color='blue', dims = [0,1,2,3,4,5,6,7],
    			truth_color='black', show_titles=True, max_n_ticks=5, quantiles=None,
    			fig=(fig, axes[:, :8]), label_kwargs = {'fontsize': 16}, title_kwargs = {'fontsize': 16}, 
    			labels = [r'$\mu_{comp}^{(1)}$', r'$\sigma \rho_{ratio,pop}^{(1)}$',  r'$\mu_{comp}^{(2)}$', 
    			r'$\sigma \rho_{ratio,pop}^{(2)}$', r'$\mu_{comp}^{(3)}$', r'$\sigma \rho_{ratio,pop}^{(3)}$', r'$\pi^{(1)}$', r'$\pi^{(2)}$'])
    
    
    
        	#mu x labels	
            axes[0,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[1,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[2,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[3,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[4,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[5,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[6,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[7,0].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[7,0].set_xticklabels([str(round(f.grids.cmf_to_wat(0.0), 2)), 
    			str(round(f.grids.cmf_to_wat(0.145), 2)), '0.3', '0.5', '0.75', '1.0'], fontsize = 16)
    
            axes[2,2].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[3,2].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[4,2].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[5,2].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[6,2].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[7,2].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[7,2].set_xticklabels([str(round(f.grids.cmf_to_wat(0.0), 2)), 
    			str(round(f.grids.cmf_to_wat(0.145), 2)), '0.3', '0.5', '0.75', '1.0'], fontsize = 16)
    
            axes[4,4].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[5,4].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[6,4].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[7,4].set_xticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[7,4].set_xticklabels([str(round(f.grids.cmf_to_wat(0.0), 2)), 
    			str(round(f.grids.cmf_to_wat(0.145), 2)), '0.3', '0.5', '0.75', '1.0'], fontsize = 16)
    
        	#mu y labels
            axes[2,0].set_yticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[2,1].set_yticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[2,0].set_yticklabels([str(round(f.grids.cmf_to_wat(0.0), 2)), 
    			str(round(f.grids.cmf_to_wat(0.145), 2)), '0.3', '0.5', '0.75', '1.0'], fontsize = 16)
    
            axes[4,0].set_yticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[4,1].set_yticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[4,2].set_yticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])
            axes[4,3].set_yticks([0.0, 0.145, 0.3, 0.5, 0.75, 1.0])	
            axes[4,0].set_yticklabels([str(round(f.grids.cmf_to_wat(0.0), 2)), 
    			str(round(f.grids.cmf_to_wat(0.145), 2)), '0.3', '0.5', '0.75', '1.0'], fontsize = 16)		
    
        	#MW y lims
            axes[6,0].set_ylim(0,1)
            axes[6,1].set_ylim(0,1)
            axes[6,2].set_ylim(0,1)
            axes[6,3].set_ylim(0,1)
            axes[6,4].set_ylim(0,1)
            axes[6,5].set_ylim(0,1)
    
            axes[7,0].set_ylim(0,1)
            axes[7,1].set_ylim(0,1)
            axes[7,2].set_ylim(0,1)
            axes[7,3].set_ylim(0,1)
            axes[7,4].set_ylim(0,1)
            axes[7,5].set_ylim(0,1)
            axes[7,6].set_ylim(0,1)
    
        	#MW x lims
            axes[6,6].set_xlim(0,1)
            axes[7,6].set_xlim(0,1)
            axes[7,7].set_xlim(0,1)
    
     
            plt.tight_layout()   
            fn = plot_path + froot + '_threepop_hyperparam_corner' + '.png'
            plt.savefig(fn) 


    def traceplot(results, froot = ''): 
        fig, axes = plt.subplots(8, 2, figsize=(10, 20))
        axes = axes.reshape((8, 2))  # reshape axes


        fig, axes = dyplot.traceplot(results, dims = [0,1, 2, 3, 4, 5, 6, 7], truth_color='black', show_titles=True,
			trace_cmap='viridis', connect=True, label_kwargs = {'fontsize': 18}, title_kwargs = {'fontsize': 18}, 
			labels = [r'$\mu_{comp}^{(1)}$', r'$\sigma \rho_{ratio,pop}^{(1)}$',  r'$\mu_{comp}^{(2)}$', r'$\sigma \rho_{ratio,pop}^{(2)}$', 
			r'$\mu_{comp}^{(3)}$', r'$\sigma \rho_{ratio,pop}^{(3)}$', r'$\pi^{(1)}$', r'$\pi^{(2)}$'],
			connect_highlight=range(5), fig=(fig, axes))  


        plt.tight_layout()
        fn = plot_path + froot + 'threepop_traceplot' + '.png'
        plt.savefig(fn)


