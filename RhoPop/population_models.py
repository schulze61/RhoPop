# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:28:21 2023
"""

import pandas as pd
import dynesty
import numpy as np
import scipy.stats as sp
import scipy.interpolate as si
from multiprocessing import cpu_count
from dynesty import NestedSampler
from dynesty import utils as dyfunc

import sys
sys.path.append('../')
import functions as f
import dynesty_plotting

from Run_rhopop import data_file

output_files_root = 'continuum'
dynesty_plots = False

#data_file = input("Type something to test this out: ")
data = np.dstack((data_file['mass'], data_file['rho_ratio']))[0]
data_err = np.dstack((data_file['mass_err'], data_file['rho_ratio_err']))[0]

#define the mass priors
planet_priors = np.array([sp.norm(loc = data[i, 0], 
                          scale = data_err[i,0]) for i in range(0, len(data))])

number_hyper_params = 2
    
def ptform(u):
        x = np.array(u)  # copy u
    	#hyperparameter prior transformation
        x[0] = 0.94*u[0]
        x[1] = 2.0*u[1]
    	#planet priors transformation
        x[number_hyper_params:] = [planet_priors[i-number_hyper_params].ppf(u[i]) for i in range(number_hyper_params, len(u))]

        return x


def loglike(x):

        X_tmp = x[number_hyper_params:]
        if x[0] >= 0.29:
            iline_c1 = f.grids.build_isoline_rock(x[0])/f.grids.build_isoline_rock(0.29)
        else:
            wmf = f.grids.cmf_to_wat(x[0])
            iline_c1 = f.grids.build_isoline_wat(wmf)/f.grids.build_isoline_rock(0.29)
        
        ii1 = si.interp1d(f.grids.get_mass_array(), iline_c1, fill_value = 'extrapolate')
    
        sigk1 = np.sqrt(data_err[:,1]**2 + x[1]**2)
    
        #first term of the log-likelihood function
        pp1 = sp.norm.pdf(data[:,1], loc = ii1(X_tmp), scale = sigk1)
        
        if len(np.where(np.isnan(pp1) == True)[0]) > 0 or len(np.where(pp1 == 0)[0]) > 0:
            return -np.inf
        else:
            pp = sum(np.log(pp1))
            
    
        #second term of the log-likelihood function
        mpp = sp.norm.pdf(data[:,0], X_tmp, data_err[:,0])
        if len(np.where(np.isnan(mpp) == True)[0]) > 0 or len(np.where(mpp == 0)[0]) > 0:
            return -np.inf
        else:
            mpp = sum(np.log(mpp))
    
        return pp + mpp


nplanets = len(data)
ndim = number_hyper_params + nplanets

colnames = np.array(['                              ']*ndim)
colnames[0:number_hyper_params] = ['c1', 'sc1']
ind = number_hyper_params
for i in range(0, len(data)):
	colnames[ind] = data_file['Planet'].iloc[i] + '_mtrue'
	ind = ind + 1
    
if __name__ == '__main__':
    with dynesty.pool.Pool(cpu_count(), loglike, ptform) as pool:
        dns = NestedSampler(loglike, ptform, ndim, pool =pool, nlive = 1000)
        #dns = NestedSampler(loglike, ptform, ndim)
        #dns.run_nested()

        results = dns.results
        print(results.summary())
    
        samples = results.samples
        weights = results.importance_weights()
        logz = results.logz
        logzerr = results.logzerr
    
    
        mean, cov = dyfunc.mean_and_cov(samples, weights)
    
        df = pd.DataFrame(samples, columns = colnames)
        df2 = pd.DataFrame({'weights': weights})
        df3 = pd.DataFrame({'logz': logz})
        df4 = pd.DataFrame({'logzerr': logzerr})
        df = df.join(df2)
        df = df.join(df3)
        df = df.join(df4)
    
        df.to_csv('results_post_' + output_files_root + '.csv') 
        
        
        
        if dynesty_plots == True:
            dynesty_plotting.continuum.runplot(results)
            dynesty_plotting.continuum.corner(results)
            dynesty_plotting.continuum.traceplot(results)

        
        
    
                
            
