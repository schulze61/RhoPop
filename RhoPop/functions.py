# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:20:26 2023

@author: jgsch
"""

import pandas as pd
import numpy as np
import scipy.interpolate as si
import scipy.optimize as so
from math import erf

rhopop_root = "./RhoPop/"
gridpath = rhopop_root + 'grids/'
MR_path = './MR_files/'

def get_density_ratios(data_file_name, overwrite_file = False):
    data_file = pd.read_csv(MR_path + data_file_name)
    data_file['rho'] = 5.51*data_file['mass']/(data_file['radius']**3)
    data_file['rho_ratio'] = data_file['rho']/grids.get_rho_scaled(data_file['mass'])
    data_file['rho_err'] = np.sqrt((5.51*data_file['mass_err'] / (data_file['radius']**3))**2 
                        + (5.51*3.0*data_file['mass']*data_file['radius_err']/(data_file['radius']**4))**2)
    data_file['rho_ratio_err'] = data_file['rho_ratio']*(data_file['rho_err']/data_file['rho'])
    if overwrite_file == True:
        data_file.to_csv(MR_path + data_file_name, index = False)
    return data_file
    

class constants:
    radius_earth_m = 6371000.0
    radius_earth_km = radius_earth_m/1000.0
    radius_earth_cm = radius_earth_m*100.0
    
    mass_earth_grams = 5.972*(10**27)
    mass_earth_kg = mass_earth_grams/1000.0


class NRPZ:
    def rho_upper(R):
        return 4.6 + 1.5*(R**2.1)

    def rho_lower(R):
        return 3.6 + 0.91*(R**2.1)
    
    
    
watgrid = pd.read_csv(gridpath+'watgrid.csv', index_col = False)
rockgrid = pd.read_csv(gridpath+'rockgrid.csv', index_col = False)

watcols = np.array([])
for i in range(1, len(watgrid.columns)):
    watcols = np.append(watcols, float(watgrid.columns[i]))


rockcols = np.array([])
for i in range(1, len(rockgrid.columns)):
    rockcols = np.append(rockcols, float(rockgrid.columns[i]))


class grids:


    def get_mass_array():
        return watgrid['mass']
    
    def get_watgrid():
        return watgrid
    def get_rockgrid():
        return rockgrid

    def build_isoline_rock(x):
        grid = rockgrid
        cols = rockcols
        d = np.sort(abs(x-cols))   
        ind1, ind2 = np.where(abs(x-cols) == d[0])[0], np.where(abs(x-cols) == d[1])[0]
        neigh = np.array([cols[ind1][0], cols[ind2][0]])
        neigh = np.sort(neigh)
    
        iline = np.zeros(len(grid))
        for i in range(0, len(grid)):
            interp = si.interp1d([neigh[0], neigh[1]], [grid[str(neigh[0])].iloc[i], grid[str(neigh[1])].iloc[i]], fill_value = 'extrapolate') 
            iline[i] = interp(x)
        
        return iline



    def build_isoline_wat(x):
        grid = watgrid
        cols = watcols
        d = np.sort(abs(x-cols))   
        ind1, ind2 = np.where(abs(x-cols) == d[0])[0], np.where(abs(x-cols) == d[1])[0]
        neigh = np.array([cols[ind1][0], cols[ind2][0]])
        neigh = np.sort(neigh)
    
        iline = np.zeros(len(grid))
        for i in range(0, len(grid)):
            interp = si.interp1d([neigh[0], neigh[1]], [grid[str(neigh[0])].iloc[i], grid[str(neigh[1])].iloc[i]], fill_value = 'extrapolate') 
            iline[i] = interp(x)
        
        return iline
    
    def calc_cmf(FeMg):
        SiMg = 0.79
        CaMg = 0.07
        AlMg = 0.09
    
        mO = 15.999
        mCa = 40.078
        mFe = 55.845
        mAl = 26.981539
        mSi = 28.0855
        mMg = 24.305
    
        num = mFe*FeMg
        mu = CaMg*(mCa + mO) + AlMg*(mAl + 1.5*mO) + SiMg*(mSi + 2.0*mO) + (mMg + mO)
        return num / (num + mu)
    
    def cmf_to_wat(x):
     return 1.0 - x*(1.0/0.29)
 
    def get_rho_scaled(M):
        iline = si.interp1d(grids.get_mass_array(), grids.build_isoline_rock(0.29))
        return iline(M)
 
    
class bayes_factor:
    def calc_bayes_factor(ZA, ZNH):
        #input needs to be ln(Z). 
        #ZNH is the likelihood corresponding to the null hypothesis.
        #ZA is the likelihood of the alternative hypothesis.
        #If the value returned is <1 then the null hypothesis is favored
        #If the value is >1 then the alternative is favored with varying degrees of significance
        return np.exp(ZA)/np.exp(ZNH)
        
    
    def calc_p_value(bayes_factor):
        def p_val_bayes_factor_expr(p, bm):
            return bm + 1.0/(np.exp(1) * p * np.log(p))
        
        return so.brentq(p_val_bayes_factor_expr, 10**(5), 10**(-9), args = (bayes_factor))
    
    def pval_to_nsigma(p_value):
        def nsigma(nsig, p):
            return p - (1.0 - erf(nsig/np.sqrt(2))) 
        return so.brentq(nsigma, 50.0, 0.0, args = (p_value))
     
    

    




    

    
    


    
