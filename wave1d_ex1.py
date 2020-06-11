"""
@author: spinosa anna, li xiao

This script intends to solve question 1 of the project
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
import matplotlib.dates as mdates 

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
    #boundary (western water level)
    #1) simple function
    s['h_left'] = 2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t)
    
    #2) read from file   # boundaries are not needed to answer the first question, thus the following 5 lines are commented before running the code
#    (bound_times,bound_values)=timeseries.read_series('tide_cadzand.txt')
#    bound_t=np.zeros(len(bound_times))
#    for i in np.arange(len(bound_times)):
#        bound_t[i]=(bound_times[i]-reftime).total_seconds()
#    s['h_left'] = np.interp(t,bound_t,bound_values)        
    return s

def timestep(x,i,settings): #return (h,u) one timestep later
    # take one timestep
    temp=x.copy() 
    A=settings['A']
    B=settings['B']
    rhs=B.dot(temp) #B*x
    rhs[0]=settings['h_left'][i] #left boundary
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
    # u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
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
# =============================================================================
def plot_series(t,series_data,s):
    # plot timeseries from model and observations
    loc_names_waterlevel =s['loc_names_waterlevel']  
    loc_names_velocity=s['loc_names_velocity']
    hseries=len(loc_names_waterlevel)
    useries=len(loc_names_velocity)
    colors = [ 'b-', 'r-', 'y-', 'g-', 'm-']
    labels = ['0', '25', '50', '75', '99']
    labels_u = ['0-25', '25-50', '50-75', '75-99']
    fig,axes=plt.subplots(2,1, figsize=(12.8,4.8),sharex=True)
    plt.suptitle('Wave1D - Model results at different stations', y=0.97,size=16)
    for i in range(hseries):
        axes[0].plot(t,series_data[i,:], colors[i], label = labels[i])
        axes[0].legend(loc='lower left')
        axes[0].set_ylabel('h [m]', size=14)
    for i in range(useries):
        axes[1].plot(t,series_data[i+hseries,:], colors[i], label = labels_u[i])
        axes[1].legend(loc='lower left')
        axes[1].set_ylabel('u [m/s]', size=14)
        axes[1].set_xlabel('time [s]', size=14)
        plt.xticks(rotation=0)
    #axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'), )
    axes[1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    axes[1].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    axes[1].xaxis.grid(False, which='minor')
    axes[1].yaxis.grid(False)
    axes[1].tick_params(axis="x", which="major", pad=12)

    plt.savefig("wave1d_ex1.png", dpi=300) 

# =============================================================================
# Print model result at different stations 
# =============================================================================
def simulate():
    # for plots
    plt.close('all')
    # locations of observations
    s=settings()
    L=s['L']
    dx=s['dx']
    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) #indices of waterlevel locations in x
    loc_names=[]
    loc_names_waterlevel = []
    loc_names_velocity = []
    names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
    for i in range(len(xlocs_waterlevel)):
        loc_names_waterlevel.append('Waterlevel at x=%f km %s'%(0.001*xlocs_waterlevel[i],names[i]))
        loc_names.append('Waterlevel at x=%f km %s'%(0.001*xlocs_waterlevel[i],names[i]))
    for i in range(len(xlocs_velocity)):
        loc_names_velocity.append('Velocity at x=%f km %s'%(0.001*xlocs_velocity[i],names[i]))
        loc_names.append('Velocity at x=%f km %s'%(0.001*xlocs_velocity[i],names[i]))
    s['loc_names_waterlevel'] = loc_names_waterlevel
    s['loc_names_velocity'] = loc_names_velocity
    s['xlocs_waterlevel']=xlocs_waterlevel
    s['xlocs_velocity']=xlocs_velocity
    s['ilocs']=ilocs
    s['loc_names']=loc_names
    
    #
    (x,t0)=initialize(s)
    t=s['t'][:] #[:40]
    times=s['times'][:] #[:40]
    series_data=np.zeros((len(ilocs),len(t)))
    for i in np.arange(1,len(t)):
        x=timestep(x,i,s)
        series_data[:,i]=x[ilocs] # x[ilocs] prints h, x[ilocs+1] prints u
        for i in np.arange(1,len(t)):
        #print('timestep %d'%i)
            x=timestep(x,i,s)
            series_data[:,i]=x[ilocs] # x[ilocs] prints h, x[ilocs+1] prints u
            
# =============================================================================
# Propagation speed as from the model. For each station and each wave, wave height peak and its time of occurance is calculated.
# This is used as t to calculate the velocity. The wave propagation speed as from the model is calculated using different x as input.            
# =============================================================================
    for k in np.arange(5):
        for j in np.arange(4):
             temp_h=series_data[k,72*j:72*(j+1)-1]
             num_h=np.max(temp_h)
             print(k,j,num_h)
             ciao_h= np.argmax(temp_h)+72*j
             print(times[ciao_h])
        
    plot_series(times,series_data,s)
    

#main program
if __name__ == "__main__":
    simulate()
#    plt.show()
