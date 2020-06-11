"""
@author: spinosa anna, li xiao

This script intends to solve question 4 of the project
"""
# 1d shallow water model
#
# solves
# dh/dt + D du/dx = 0
# du/dt + g dh/dx + f*u = 0
# 
# staggered discretiztation in space and central in time
#
# o -> o -> o -> o ->   # staggering
# L u  h u  h u  h  R   # element
# 0 1  2 3  4 5  6  7   # index in state vector
#
# m=1/2, 3/2, ...
#  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
#= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
# m=1,2,3,...
#  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
#= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import timeseries
import dateutil 
import datetime
import math
import matplotlib.dates as mdates 
from sklearn.metrics import mean_squared_error
from math import sqrt

minutes_to_seconds=60.
hours_to_seconds=60.*60.
days_to_seconds=24.*60.*60.

def settings():
    s=dict() #hashmap to  use s['g'] as s.g in matlab
    # Constants
    s['g']=9.81 # acceleration of gravity
    s['D']=20.0 # Depth
    s['f']=1/(0.06*days_to_seconds) # damping time scale
    L=100.e3 # length of the estuary
    s['L']=L
    n=100 #number of cells
    s['n']=n    
    # Grid(staggered water levels at 0 (boundary) dx 2dx ... (n-1)dx
    #      velocities at dx/2, 3dx/2, (n-1/2)dx
    dx=L/(n+0.5)
    s['dx']=dx
    x_h = np.linspace(0,L-dx,n)
    s['x_h'] = x_h
    s['x_u'] = x_h+0.5    
    # initial condition
    s['h_0'] = np.zeros(n)
    s['u_0'] = np.zeros(n)    
    # time
    t_f=2.*days_to_seconds #end of simulation
    dt=10.*minutes_to_seconds
    s['dt']=dt
    reftime=dateutil.parser.parse("201312050000") #times in secs relative
    s['reftime']=reftime
    t=dt*np.arange(np.round(t_f/dt))
    s['t']=t
# =============================================================================
# boundary (western water level). Tide data given at Cadnaz are employed to define the left boundary.
# data are read from .txt file 
# =============================================================================  
    (bound_times,bound_values)=timeseries.read_series('tide_cadzand.txt')
    bound_t=np.zeros(len(bound_times))
# =============================================================================
# AutoRegression - Sthocastic forcing is introduced to the left boundary
# =============================================================================
    ensemble_number=50
    T=6.*hours_to_seconds
    alpha=np.exp(-dt/T)
    #time=bound_t.shape[0]
    time=len(t)
    theta_n=0.2
    theta_w=theta_n*math.sqrt(1.0-alpha**2)
  
    n_ar=np.zeros((time,ensemble_number))
    w_ar=np.zeros((time,ensemble_number))
    bound_values_ensemble=np.zeros((time,ensemble_number))
    n_ar,w_ar=west_boundary_AR(theta_w,theta_n,alpha,ensemble_number,t)
    for j in np.arange(ensemble_number):
        for i in np.arange(len(bound_times)):
            bound_t[i]=(bound_times[i]-reftime).total_seconds()
        temp= np.interp(t,bound_t,bound_values)   
        bound_values_ensemble[:,j]=temp +n_ar[:,j]   
    s['h_left'] = bound_values_ensemble
#######################This part if for AR(1)####
      
    return s

def west_boundary_AR(theta_w,theta_n,alpha,ensemble_number,t):
    N=np.zeros((len(t),ensemble_number))
    W=np.zeros((len(t),ensemble_number))
    N[0,:]=np.random.normal(0,theta_n,ensemble_number)    
    for i in range(1,len(t),1):
        W[i,:]=np.random.normal(0,theta_w,ensemble_number)   
        N[i,:]=alpha*N[i-1,:]+W[i,:]
    return N,W

def timestep(x,i,j,settings): #return (h,u) one timestep later
    # take one timestep
    temp=x.copy() 
    A=settings['A']
    B=settings['B']
    rhs=B.dot(temp) #B*x
    rhs[0]=settings['h_left'][i,j] #left boundary
    newx=spsolve(A,rhs)
    return newx

def initialize(settings): #return (h,u,t) at initial time 
    #compute initial fields and cache some things for speed
    h_0=settings['h_0']
    u_0=settings['u_0']
    n=settings['n']
    x=np.zeros(2*n) #order h[0],u[0],...h[n],u[n]
    x[0::2]=u_0[:]
    x[1::2]=h_0[:]
    #time
    t=settings['t']
    reftime=settings['reftime']
    dt=settings['dt']
    times=[]
    second=datetime.timedelta(seconds=1)
    for i in np.arange(len(t)):
        times.append(reftime+i*int(dt)*second)
    settings['times']=times
    #initialize coefficients
    # create matrices in form A*x_new=B*x+alpha 
    # A and B are tri-diagonal sparse matrices 
    Adata=np.zeros((3,2*n)) #order h[0],u[0],...h[n],u[n]  
    Bdata=np.zeros((3,2*n))
    #left boundary
    Adata[1,0]=1.
    #right boundary
    Adata[1,2*n-1]=1.
    # i=1,3,5,... du/dt  + g dh/sx + f u = 0
    #  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
    #= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
    g=settings['g'];dx=settings['dx'];f=settings['f']
    temp1=0.5*g*dt/dx
    temp2=0.5*f*dt
    for i in np.arange(1,2*n-1,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0 + temp2
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0 - temp2
        Bdata[2,i+1]= -temp1
    # i=2,4,6,... dh/dt + D du/dx = 0
    #  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
    #= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])
    D=settings['D']
    temp1=0.5*D*dt/dx
    for i in np.arange(2,2*n,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0
        Bdata[2,i+1]= -temp1    
    # build sparse matrix
    A=spdiags(Adata,np.array([-1,0,1]),2*n,2*n)
    B=spdiags(Bdata,np.array([-1,0,1]),2*n,2*n)
    A=A.tocsr()
    B=B.tocsr()
    settings['A']=A #cache for later use
    settings['B']=B
    return (x,t[0])


# =============================================================================
# Plot model oucomes of wave height (h [m]) and velocity (u [m/s]) at different stations 
# Plot observed tidal data 
# Plot the 95% confidence interval 
# =============================================================================
        
def plot_series(t,series_data,confidence_top,confidence_bottom,s,obs_data):
    # plot timeseries from model and observations
    loc_names_waterlevel =s['loc_names_waterlevel']  
    loc_names_velocity=s['loc_names_velocity']
    hseries=len(loc_names_waterlevel)
    useries=len(loc_names_velocity)
    for i in range(hseries):
        fig,ax=plt.subplots()
        ax.plot(t,series_data[i,:],'r-', linewidth=1, label = 'ensemble mean')
        #ax.plot(t,confidence_top[i,:],'r-', label ='95 % confidence interval')
        #ax.plot(t,confidence_bottom[i,:],'r-')
        ax.fill_between(t, confidence_top[i,:], confidence_bottom[i,:], label ='95 % confidence interval', color='skyblue')
        ntimes=min(len(t),obs_data.shape[1])
        ax.plot(t[0:ntimes],obs_data[i,0:ntimes],':', color='black', label ='observations')
        ax.set_ylim(-3.5,3.5)
        #ax.set_xlabel('time')
        ax.set_ylabel('h [m]', size=12)
        ax.set_title(loc_names_waterlevel[i],size =12)#, x=0.05, y=0, )
        #plt.legend()
        ax.legend(loc='lower right', fontsize=10)
        
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'), )
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.grid(False, which='minor')
        ax.yaxis.grid(False)
        ax.tick_params(axis="x", which="major", size=12)
    
        plt.savefig(("ex4_%s.png"%loc_names_waterlevel[i]).replace(' ','_'))
        
    # =============================================================================
    # Compute some statistic to gain a first quantitative idea of the accuracy of the model.
    # =============================================================================
        error=-series_data[i,:]+obs_data[i,0:ntimes]    
        rmse = sqrt(mean_squared_error(obs_data[i,0:ntimes], series_data[i,:]))
        print(i, 'RMSE'+str(rmse))
        print(i, 'MAE'+str(sum(abs(error))/288.0)) #MAE
        print(i, 'Mean'+str(np.mean(error))) 
        print(i, 'Std'+str(np.std(error)))
        print(i, 'MIn'+str(np.min(error)))
        print(i, 'Max'+str(np.max(error)))
        
    for i in range(useries):
        fig,ax=plt.subplots()
        ax.plot(t,series_data[i+hseries,:],'r-', linewidth=1, label = 'ensemble mean')
        #ax.plot(t,confidence_top[i,:],'r-', label ='95 % confidence interval')
        #ax.plot(t,confidence_bottom[i,:],'r-')
        ax.fill_between(t, confidence_top[i+hseries,:], confidence_bottom[i+hseries,:], label ='95 % confidence interval', color='C9')
        ax.set_ylim(-2.5,2.5)
        #ax.set_xlabel('time')
        ax.set_ylabel('u [m/s]', size=12)
        ax.set_title(loc_names_velocity[i], size=12) #, x=0.05, y=0, size=12)
        #plt.legend()       
        
        ax.legend(loc='lower right', fontsize=10)
        
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'), )
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.grid(False, which='minor')
        ax.yaxis.grid(False)
        ax.tick_params(axis="x", which="major", size=12)
    
        plt.savefig(("ex4_%s.png"%loc_names_velocity[i]).replace(' ','_'))

    
def simulate():
    # for plots
    plt.close('all')
    fig1,ax1 = plt.subplots() #maps: all state vars at one time
    # locations of observations
    s=settings()
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) #indices of waterlevel locations in x
    loc_names=[]
    loc_names_velocity =[]
    loc_names_waterlevel=[]
    names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
    for i in range(len(xlocs_waterlevel)):
        loc_names_waterlevel.append(('Waterlevel at x=%f km %s'%(0.001*xlocs_waterlevel[i],names[i])))
        loc_names.append('Waterlevel at x=%f km %s'%(0.001*xlocs_waterlevel[i],names[i]))
    for i in range(len(xlocs_velocity)):
        loc_names_velocity.append('Velocity at x=%f km %s'%(0.001*xlocs_velocity[i],names[i]))
        loc_names.append('Velocity at x=%f km %s'%(0.001*xlocs_velocity[i],names[i]))
    s['xlocs_waterlevel']=xlocs_waterlevel
    s['xlocs_velocity']=xlocs_velocity
    s['ilocs']=ilocs
    s['loc_names']=loc_names
    s['loc_names_velocity'] = loc_names_velocity
    s['loc_names_waterlevel'] = loc_names_waterlevel
    #
    (x,t0)=initialize(s)
    t=s['t'][:] #[:40]
    times=s['times'][:] #[:40]
    series_data=np.zeros((len(ilocs),len(t)))
    N=50 #### ensemble size for AR(1)
    x_model=np.zeros((N,len(ilocs),len(t)))
    for j in np.arange(0,N):
       (x,t0)=initialize(s) 
       for i in np.arange(1,len(t)):
        #print('timestep %d'%i)
        
          x=timestep(x,i,j,s)
        #plot_state(fig1,x,i,s) #show spatial plot; nice but slow
          series_data[:,i]=x[ilocs]
       x_model[j,:,:]=series_data  
# =============================================================================
# Calculate ensemble mean and confidence interval
# =============================================================================
    series_data=np.mean(x_model,axis=0)   
    model_var=np.std(x_model,axis=0)
    confidence_temp=1.96*model_var/math.sqrt(N)
    confidence_top=series_data+confidence_temp
    confidence_bottom=series_data-confidence_temp
    
    #load observations
    (obs_times,obs_values)=timeseries.read_series('tide_cadzand.txt')
    observed_data=np.zeros((len(ilocs),len(obs_times)))
    observed_data[0,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('tide_vlissingen.txt')
    observed_data[1,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('tide_terneuzen.txt')
    observed_data[2,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('tide_hansweert.txt')
    observed_data[3,:]=obs_values[:]
    (obs_times,obs_values)=timeseries.read_series('tide_bath.txt')
    observed_data[4,:]=obs_values[:]
    plot_series(times,series_data,confidence_top,confidence_bottom,s,observed_data)
    #plot_series(times,series_data,s,observed_data)

#main program
if __name__ == "__main__":
    simulate()
    #plt.show()
