# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:33:25 2020

@author: spinosa
"""

import wave1d
import timeseries
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import errno
from scipy.sparse.linalg import inv
from scipy.sparse import hstack, vstack, csc_matrix
from numpy import loadtxt
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates  


def get_globals(module_name):
    module = globals().get(module_name, None)
    variables = {}
    if module:
        variables = {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
        #print(variables)
    return variables

wave1d_globals = get_globals('wave1d')
#print(wave1d_globals)


minutes_to_seconds=60.
dt=10.*minutes_to_seconds
days_to_seconds=24.*60.*60.

def settings_enkf(func):
    def wrapper(n_ensembles,sigma_v,*args, **kwargs):
        s = func()
        # W_sigma
        T = 6 * wave1d_globals['hours_to_seconds']
        alpha = np.exp(-s['dt'] / T)
        s['alpha'] = alpha
        sigma_n = 0.2
        sigma_w = sigma_n * np.sqrt(1 - alpha ** 2)
        s['sigma_w'] = sigma_w
        #Emsembles
        n_ensembles = n_ensembles
        s['n_ensembles'] = n_ensembles
        W = np.random.randn(len(s['h_left']), n_ensembles) * sigma_w
        s['W'] = W
        #hleft emsemble matrix creation
        bound_values_old = s['h_left']
        bound_values_matrix = np.array(bound_values_old) * np.ones((len(bound_values_old), n_ensembles)).T
        #print(bound_values_matrix)
        s['h_left_ensembles'] = bound_values_matrix #+ noise_ar_1
        # Observations
        obs_index = [50, 100, 150, 198] #from ilocs
        s['obs_index'] = obs_index
        #H matrix
        row = np.array(range(len(obs_index)))
        col = np.array(obs_index)
        data = np.ones(len(obs_index))
        H = coo_matrix((data, (row, col)), shape=(len(obs_index),2*s['n']+1)).toarray()
        s['H'] = H
        s['sigma_v'] = sigma_v
        return s
    return wrapper

# =============================================================================
# Model extention 
# =============================================================================

def initialize_enkf(func):
    def wrapper(s,*args, **kwargs):
        x, t0 = func(s,*args, **kwargs)
        #print(x)
        # ORIGINAL MODEL - A xk+1 = B xk
        # EXTENDED MODEL- xk+1= M xk + C bk + wk
        # Extending X
        x = np.append(x,[0])
        A = s['A'].tocsc()
        B = s['B'].tocsc()
        C = inv(A)
        M = C.dot(B)
        # Extending M and C
        M_new = M.copy()
        M_new.resize((201, 201))
        M_new[0, 2*s['n']] = 1
        M_new[2*s['n'], 2*s['n']] = s['alpha']
        C_new = C.copy()
        C_new.resize((2*s['n']+1,2*s['n']+1))
        s['M_ext'] = M_new
        s['C_ext'] = C_new
        # Creating X0 Ensemble
        xi_0 = x * np.ones((s['n_ensembles'], len(x)))
        return (xi_0,t0)
    return wrapper

# =============================================================================
# EnKF implementation: for each time step, boundary vector and noise are creaed and a new state vector is returned
# =============================================================================
    
def enkf_impl(x,t,s,ensemble_index):
    # Create boundary condition vector [h_b(k), 0, 0, ... 0]
    h_b = np.zeros(2 * s['n'] + 1)
    h_b[0] = s['h_left'][t]
    # Create noise vector [0, 0, 0, ... w(k)]
    w = np.zeros(2 * s['n'] + 1)
    w[-1] = s['W'][t][ensemble_index]
    M_new = s['M_ext']
    C_new = s['C_ext']
    # Perform model timestep
    new_x = M_new.dot(x) + C_new.dot(h_b) + w
    return new_x

# =============================================================================
# For each new_x a forecast (x_f) is generated. The state xi-1 defined for the previous state is used as input
# The forecast is returned for each time
# The mean of the ensemble is also calculated and returned
# =============================================================================
def ensembles_of_forecast(xi_1, s, t):
   # xk+1= M xk + C bk + wk
    xi_f = [enkf_impl(xi_1[i],t,s,i) for i in range(len(xi_1))]
    # Ensemble estimate
    x_f = np.mean(xi_f,axis=0)
    return xi_f,x_f

# =============================================================================
# Calculate covariance of forecast and forecast erro. These will be used to calculate the Kalman gain
# =============================================================================

def Cov_Lf(xi_f,x_f,H,N):    
    error = xi_f - x_f
    L_f = np.array(error)/ np.sqrt(N - 1)
    psi = np.inner(L_f, H)
    return L_f ,psi
                  
# =============================================================================
# Kalman gain
# =============================================================================
def K_gain(L_f,psi,R):
    aux1 = np.inner(psi.T,L_f.T)
    aux2 = np.linalg.inv(np.inner(psi.T,psi.T) + R)
    K = np.inner(aux2,aux1.T)
    return K

# =============================================================================
# Create observation noise
# =============================================================================
def observations_noise(s):
    v = []
    n_ensembles = s['n_ensembles'] # will be defined in settings
    sigma_v = s['sigma_v'] # will be defined in settings
    for j in range(len(s['obs_index'])): # Change later n_ensembles b
        v.append(np.random.normal(0, sigma_v, n_ensembles))
    v = np.array(v).T
    return v

# =============================================================================
# Assimilation. State estimate and assimilated ensemble are returned
# =============================================================================
def assimilate(xi_f, z_k, H, K, v):
    aux0 = np.inner(xi_f, H)
    aux1 = z_k - aux0 + v
    aux2 = np.inner(aux1, K.T)
    assimilated = xi_f + aux2
    x_e = np.mean(assimilated, axis=0)
    return assimilated, x_e
        

# =============================================================================
# Plot model outcomes of wave height (h [m]) and velocity (u [m/s]) at different stations 
# Plot observed data. Those are the observed waveheigh
# Plot the assimilated values.
# =============================================================================

def plot_series(t,series_data,s,obs_data,series_model):
  #   plot timeseries from model and observations
    loc_names=s['loc_names']
    for i in range(5):
        fig,ax=plt.subplots()
        fmodel = loadtxt('model.csv', delimiter =',')
        ax.plot(t,fmodel[i,:],'g-', label = 'model')
        ntimes=min(len(t),obs_data.shape[1])
        ax.plot(t[0:ntimes],obs_data[i,0:ntimes],'r-', label ='observations')
        ax.plot(t,series_data[i,:],'b-', label ='assimilated')
        ax.set_ylim(-3.5,5)
        #ax.set_xlabel('time')
        ax.set_ylabel('h [m]', size=12)
        ax.set_title(loc_names[i],size =12)#, x=0.05, y=0, )
        #plt.legend()
        ax.legend(loc='lower left', fontsize=10)
        
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'), )
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.grid(False, which='minor')
        ax.yaxis.grid(False)
        ax.tick_params(axis="x", which="major", size=12)
    
        plt.savefig(("ex9_%s.png"%loc_names[i]).replace(' ','_'))
        
    # =============================================================================
    # Compute some statistic to gain a first quantitative idea of the accuracy of the model.
    # =============================================================================
        error=-series_data[i,:]+obs_data[i,0:ntimes]    
        rmse = sqrt(mean_squared_error(obs_data[i,0:ntimes],  series_data[i,:]))

        print('&', 'Vlissigen', '&', 'Terneuzen', '&', 'Hansweert', '&', 'Bath')
        print('RMSE'+str(rmse))
        print('MAE'+str(sum(abs(error))/288.0)) #MAE
        print('Mean'+str(np.mean(error))) 
        print('Std'+str(np.std(error)))
        print('MIn'+str(np.min(error)))
        print('Max'+str(np.max(error)))
        list_u = [5,6,7,8]
    for i in list_u:
        fig,ax=plt.subplots()
        ax.plot(t,fmodel[i,:],'g-', label = 'model')
        ax.plot(t,series_data[i,:],'b-', label ='assimilated')
        ax.set_ylim(-2.5,2.5)
        #ax.set_xlabel('time')
        ax.set_ylabel('u [m/s]', size=12)
        ax.set_title(loc_names[i], size=12) #, x=0.05, y=0, size=12)
        
        ax.legend(loc='lower left', fontsize=10)
        
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'), )
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.grid(False, which='minor')
        ax.yaxis.grid(False)
        ax.tick_params(axis="x", which="major", size=12)
    
        plt.savefig(("ex9_%s.png"%loc_names[i]).replace(' ','_'))
         
# =============================================================================
# For each time step and ensemble it is important to save the estimated and forecast valued. 
# This function returns the variables to be saved
# =============================================================================
def saving_function(value):
    choises = {1:['series_data','x_f','x_e','xi_f','ensembles_e','K']}
    return choises[value]


def simulate(s,xi_0,t_0,observed_data,plotting=True,saving_option=3):
    savings_choices = saving_function(saving_option)
    t = s['t']
    ilocs= s['ilocs']
    series_data = np.zeros((len(s['ilocs']), len(t)))
    series_model= np.zeros((len(s['ilocs']), len(t)))
    result = {}
    #print(x)
    for i in np.arange(0,len(t)):
    # Forecast step
        xi_f, x_f = ensembles_of_forecast(xi_0, s, i);
        H = s['H'] # n_obs x len(x)
        L_f, psi = Cov_Lf(xi_f, x_f, H, len(x_f))
        # Assimilation step
        v = observations_noise(s)
        R = np.diag(np.array([s['sigma_v']**2 for i in range(len(s['obs_index']))])) # observation covariance matrix. Obs noise is assumeded to be uncorrelated 
        psi = np.inner(L_f, H)
        K = K_gain(L_f, psi, R)
        z_k = observed_data.T[i][1:len(s['obs_index'])+1]
        ensembles_e, x_e = assimilate(xi_f, z_k, H, K, v) 
        xi_0 = ensembles_e         
        series_data[:, i] = x_e[s['ilocs']]
       
        local = locals()
        result[i] = {variable: local.get(variable) for variable in savings_choices}
    
    times = s['times'][:]
   # print(len(times),np.shape(series_data),np.shape(observed_data))
    plot_series(times, series_data, s, observed_data,series_model)
    print('Final result')
    return result


def simulator(n_ensembles,saving_option, sigma_v, plotting):
    global new_settings
    global new_initialize
    global new_timestep
    # Extending wave1d functions
    new_settings = settings_enkf(wave1d.settings)
    new_initialize = initialize_enkf(wave1d.initialize)
    new_timestep = enkf_impl
    s = new_settings(n_ensembles,sigma_v)
    xi_0, t_0 = new_initialize(s)
    observed_data = wave1d.simulate()
    result = simulate(s,xi_0,t_0,observed_data, plotting=plotting, saving_option=saving_option)
    return result

# =============================================================================
# Define ensemble size, saving option, error and run the code
# =============================================================================
n_ensembles = 100
saving_option = 1
plotting =False
sigma_v= 0.1
results = simulator(n_ensembles,saving_option,sigma_v,plotting)
