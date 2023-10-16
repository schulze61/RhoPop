# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 18:25:32 2023
"""

import pandas as pd
import dynesty
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import scipy.interpolate as si
from matplotlib.ticker import MultipleLocator
import sys
sys.path.append('./RhoPop')
import functions as f

data_file = 'Fake_20planet_sample'
file_ext = '.csv'

plot_path = './output_plots/'

#raw planet data
file = data_file + file_ext
data = f.get_density_ratios(file, overwrite_file = True)

rho_scaled = f.grids.build_isoline_rock(0.29)
mass = f.grids.get_mass_array()

#Set up 1-population model results for plotting
continuum_results = './results/' + data_file + '_continuum' + file_ext
one_pop_model = pd.read_csv(continuum_results)
weights = one_pop_model['weights']
d = one_pop_model[one_pop_model.columns[0:2]]

ind_test = np.argmax(one_pop_model['logz'])


med_1pop = np.array([dynesty.utils.quantile(d['c1'], [0.5], weights=weights)[0], 
                      dynesty.utils.quantile(d['sc1'], [0.5], weights=weights)[0]])


if med_1pop[0] > 0.29:
    isocomp_1pop = f.grids.build_isoline_rock(med_1pop[0])
else:
    wmf_1pop = f.grids.cmf_to_wat(med_1pop[0])
    isocomp_1pop = f.grids.build_isoline_wat(wmf_1pop)
    
ratio_1pop = isocomp_1pop/rho_scaled
intrinsic_scatter_1pop = med_1pop[1]


#set up 2-population model for plotting
twopop_results = './results/' + data_file + '_two_pop' + file_ext
two_pop_model = pd.read_csv(twopop_results)
weights = two_pop_model['weights']
d = two_pop_model[two_pop_model.columns[0:5]]


med_2pop = np.array([dynesty.utils.quantile(d['c1'], [0.5], weights=weights)[0], 
                      dynesty.utils.quantile(d['sc1'], [0.5], weights=weights)[0],
                      dynesty.utils.quantile(d['c2'], [0.5], weights=weights)[0],
                      dynesty.utils.quantile(d['sc2'], [0.5], weights=weights)[0],
                      dynesty.utils.quantile(d['mw'], [0.5], weights=weights)[0]])


if med_2pop[0] >= 0.29:
    isocomp_2pop_higher_rho_pop = f.grids.build_isoline_rock(med_2pop[0])
else:
    wmf_2pop_higher_rho_pop = f.grids.cmf_to_wat(med_2pop[0])
    isocomp_2pop_higher_rho_pop = f.grids.build_isoline_wat(wmf_2pop_higher_rho_pop)

intrinsic_scatter_2pop_higher_rho_pop = med_2pop[1]

    
if med_2pop[2] >= 0.29:
    isocomp_2pop_lower_rho_pop = f.grids.build_isoline_rock(med_2pop[2])
else:
    wmf_2pop_lower_rho_pop = f.grids.cmf_to_wat(med_2pop[2])
    isocomp_2pop_lower_rho_pop = f.grids.build_isoline_wat(wmf_2pop_lower_rho_pop)
        
intrinsic_scatter_2pop_lower_rho_pop = med_2pop[3]


isocomp_2pop_higher_rho_pop = isocomp_2pop_higher_rho_pop/rho_scaled
isocomp_2pop_lower_rho_pop = isocomp_2pop_lower_rho_pop/rho_scaled
mw_2pop = med_2pop[4]


sigk1_2pop = np.sqrt(data['rho_ratio_err']**2 + intrinsic_scatter_2pop_higher_rho_pop**2)
sigk2_2pop = np.sqrt(data['rho_ratio_err']**2 + intrinsic_scatter_2pop_lower_rho_pop**2)


ii1 = si.interp1d(mass, isocomp_2pop_higher_rho_pop, fill_value = 'extrapolate')
ii2 = si.interp1d(mass, isocomp_2pop_lower_rho_pop, fill_value = 'extrapolate')
    
pp1 = mw_2pop*sp.norm.pdf(data['rho_ratio'], loc = ii1(data['mass']), scale = sigk1_2pop)
pp2 = (1.0-mw_2pop)*sp.norm.pdf(data['rho_ratio'], loc = ii2(data['mass']), scale = sigk2_2pop)




#assign labels
pl_labels_2pop = np.zeros(len(data))

for i in range(0, len(pl_labels_2pop)):
    if pp1[i]/pp2[i] > 1.0:
        pl_labels_2pop[i] = 0
    else:
        pl_labels_2pop[i] = 1
        
ind_upper_rho_pop_2pop = np.where(pl_labels_2pop == 0)[0]
ind_lower_rho_pop_2pop = np.where(pl_labels_2pop == 1)[0]



#Now do the plotting
markersize = 5
linewidth = 3
fill_alpha = 0.1


#set up figure and grid
markersize = 10
plt.figure(figsize = (20, 8))
plt.tight_layout()

ax1 = plt.subplot2grid((1,4), (0,0), colspan = 2, rowspan = 1)
ax3 = plt.subplot2grid((1,4), (0,2), colspan = 2, rowspan = 1)
plt.subplots_adjust(wspace=0.1, hspace=0)


#----------------------------------1-population results in terms of rho ratio and mass-------------------------------------#
ax1.plot(mass, ratio_1pop, 'g-', lw = linewidth, label = 'CMF = ' + str(round(med_1pop[0],2)))
ax1.errorbar(data['mass'], data['rho_ratio'], yerr = data['rho_ratio_err'], xerr = data['mass_err'], 
                 linewidth = 0, elinewidth = 1, marker = 'o', markersize = markersize, color = 'g')
ax1.fill_between(mass, ratio_1pop + intrinsic_scatter_1pop, 
                 ratio_1pop - intrinsic_scatter_1pop, color = 'g', alpha = fill_alpha)

ax1.set_ylim(0.25, 2.0)
ax1.set_xlim(0.05,10)

ax1.tick_params(which = 'major', direction = 'in', length = 15, top = True, right = True, labelsize = 24, width = 3)
ax1.tick_params(which = 'minor', direction = 'in', length = 7.5, top = True, right = True, width = 3)

ax1.patch.set_edgecolor('black')  
ax1.patch.set_linewidth('3')  

ax1.xaxis.set_major_locator(MultipleLocator(1.0))
ax1.xaxis.set_major_formatter('{x:.0f}')
ax1.xaxis.set_minor_locator(MultipleLocator(0.25))

ax1.yaxis.set_major_locator(MultipleLocator(0.5))
ax1.yaxis.set_major_formatter('{x:.1f}')
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

ax1.set_ylabel(r'$\rho / \rho_{scaled}$', fontsize = 32)
ax1.legend(fontsize = 18, frameon = False, borderpad = 1, loc = 'lower right')

#----------------------------------2-population results in terms of rho ratio and mass-------------------------------------#
ax3.plot(mass, isocomp_2pop_higher_rho_pop, 'g-', lw = linewidth, label = 'CMF = ' + str(round(med_2pop[0],2)))
ax3.fill_between(mass, isocomp_2pop_higher_rho_pop + intrinsic_scatter_2pop_higher_rho_pop, 
                 isocomp_2pop_higher_rho_pop - intrinsic_scatter_2pop_higher_rho_pop, color = 'g', alpha = fill_alpha)

ax3.plot(mass, isocomp_2pop_lower_rho_pop, '-', color = 'mediumorchid', lw = linewidth, label = 'CMF = ' + str(round(med_2pop[2],2)))
ax3.fill_between(mass, isocomp_2pop_lower_rho_pop + intrinsic_scatter_2pop_lower_rho_pop, 
                 isocomp_2pop_lower_rho_pop - intrinsic_scatter_2pop_lower_rho_pop, color = 'mediumorchid', alpha = fill_alpha)

ax3.errorbar(data['mass'].iloc[ind_upper_rho_pop_2pop], data['rho_ratio'].iloc[ind_upper_rho_pop_2pop],
             yerr = data['rho_ratio_err'].iloc[ind_upper_rho_pop_2pop], xerr = data['mass_err'].iloc[ind_upper_rho_pop_2pop], 
                 linewidth = 0, elinewidth = 1, marker = 'o', markersize = markersize, color = 'g')
ax3.errorbar(data['mass'].iloc[ind_lower_rho_pop_2pop], data['rho_ratio'].iloc[ind_lower_rho_pop_2pop],
             yerr = data['rho_ratio_err'].iloc[ind_lower_rho_pop_2pop], xerr = data['mass_err'].iloc[ind_lower_rho_pop_2pop], 
                 linewidth = 0, elinewidth = 1, marker = 'o', markersize = markersize, color = 'mediumorchid')


ax3.set_ylim(0.25, 2.0)
ax3.set_xlim(0.05,10)

ax3.tick_params(which = 'major', direction = 'in', length = 15, top = True, right = True, labelsize = 24, width = 3)
ax3.tick_params(which = 'minor', direction = 'in', length = 7.5, top = True, right = True, width = 3)

ax3.patch.set_edgecolor('black')  
ax3.patch.set_linewidth('3')  

ax3.xaxis.set_major_locator(MultipleLocator(1.0))
ax3.xaxis.set_major_formatter('{x:.0f}')
ax3.xaxis.set_minor_locator(MultipleLocator(0.25))

ax3.yaxis.set_major_locator(MultipleLocator(0.5))
ax3.yaxis.set_major_formatter('{x:.1f}')
ax3.yaxis.set_minor_locator(MultipleLocator(0.1))

ax3.axes.yaxis.set_ticklabels([])
ax3.set_xlabel(r'Mass $[M_\oplus]$', fontsize = 32)


ax1.set_xlabel(r'Mass $[M_\oplus]$', fontsize = 32)


ax3.legend(fontsize = 18, frameon = False, borderpad = 1., loc = 'lower right')

ax1.set_title(r'$N_c = 1$', fontsize = 32, pad = 20)
ax3.set_title(r'$N_c = 2$', fontsize = 32, pad = 20)

figpath = plot_path + 'summary_plot_' + file + '.png'
plt.savefig(figpath)







