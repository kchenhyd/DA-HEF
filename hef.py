
import numpy as np 
import pandas as pd
from src.da_methods import *
from src.pkg import *
from src.util import *
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

#configure ES-MDA
nreaz = 100 # number of realizations
niter = 4 # number of iterations
alpha = np.array([4.0, 4.0, 4.0, 4.0]) # iteration coefficient for each iteration
da_para = ['flux'] # parameters to be estimated
da_timestep = 7200.0 # unit: sec, data assimilation time step


flux_mean = 0.0          # unit: m/day, mean flux
flux_sd = 0.5          # unit: m/day, standard deviation of flux       
flux_up_bound = 5.0     # unit: m/day, maximum downwelling flux
flux_low_bound = -5.0   # unit: m/day, maximum upwelling flux

# Configure observation 
obs_length = 32     # unit:day, length of observation data used for flux estimation
obs_timestep = 300.0              # unit:s, the time interval that temperatures are collected
therm_loc = [-0.01, -0.05, -0.15, -0.25, -0.65] # unit:m, location of thermistor, negative means below the riverbed
obs_error_type = 'absolute'    # 'absolute' and 'relative'. 'absolute' means the absolute measurement error in degree C, 'relative' means a perentage of the observation value
obs_error = 0.05              # If the error type is 'absolute', the error means the accuracy of temperature measurement with unit degree C. If the error type is 'relative', the error means the percentage of temperature measurement.
# Configure model domain and PFLOTRAN running environment
hz = 0.64          # unit: m, height of the 1-D column
spinup_length = 0 #unit: day, spinup time

exeprg = '/home/chen454/petsc/pflotran/src/pflotran/pflotran'
mpirun = '/home/chen454/petsc/arch-linux2-c-opt/bin/mpirun'
ncore = 50

#----------------------------------------------------------
kwargs1 = {}
if 'permeability' in da_para:
    kwargs1.update({'logperm_mean':logperm_mean})
    kwargs1.update({'logperm_sd':logperm_sd})
    kwargs1.update({'logperm_low_bound':logperm_low_bound})
    kwargs1.update({'logperm_up_bound':logperm_up_bound})                 

if 'flux' in da_para:
    kwargs1.update({'flux_mean':flux_mean})
    kwargs1.update({'flux_sd':flux_sd})
    kwargs1.update({'flux_low_bound':flux_low_bound})
    kwargs1.update({'flux_up_bound':flux_up_bound}) 

if 'thermal conductivity' in da_para:
    kwargs1.update({'th_cond_mean':th_cond_mean})
    kwargs1.update({'th_cond_sd':th_cond_sd})
    kwargs1.update({'th_cond_low_bound':th_cond_low_bound})
    kwargs1.update({'th_cond_up_bound':th_cond_up_bound})

if 'porosity' in da_para:
    kwargs1.update({'poro_mean':poro_mean})
    kwargs1.update({'poro_sd':poro_sd})
    kwargs1.update({'poro_low_bound':poro_low_bound})
    kwargs1.update({'poro_up_bound':poro_up_bound})

spinup_length_sec = spinup_length*86400
obs_length_sec = obs_length*86400
obs_start_time = spinup_length_sec
obs_end_time = obs_length_sec

# create assimilation object
th1d = TH1D(da_para,nreaz,hz,spinup_length,**kwargs1)

if 'permeability' in da_para:
    da_time_win = np.array([[obs_start_time,obs_end_time]])
else:
    da_time_win = np.array([[obs_start_time,obs_start_time+da_timestep]])
    time = obs_start_time+da_timestep
    while time <= obs_end_time:
        da_time_win = np.append(da_time_win,[[time,time+da_timestep]],axis=0)
        time = time + da_timestep
    da_time_win = da_time_win[0:-1,:]

# create observation object
obs_coord = np.array(therm_loc[1:-1])-np.array(therm_loc[0])
obs_data = np.loadtxt('./observation/obs_data.dat',skiprows=1)
obs_start_idx = int(obs_start_time/obs_timestep)
obs_end_idx = int(obs_end_time/obs_timestep)
obs_data = obs_data[obs_start_idx:obs_end_idx+1,:]
obs = Obs(obs_start_time,obs_end_time,obs_timestep,obs_coord,obs_error_type,obs_error,obs_data)

# generate initial condition
temp_top =  np.loadtxt('./pflotran_inputs/temp_top.dat',skiprows=1)
temp_bot =  np.loadtxt('./pflotran_inputs/temp_bottom.dat',skiprows=1)
ncell = int(hz/0.01)
cellid = np.linspace(1,ncell,num=ncell,dtype='int')
init_temp = np.linspace(temp_bot[0,1],temp_top[0,1],num=ncell,dtype='float')

with h5py.File('./pflotran_results/init_temp.h5','w') as f:
    f.create_dataset('Cell Ids',data=cellid)
    f.create_dataset('Temperature_C',data=init_temp)
f.close()  

subprocess.call("cp ./pflotran_inputs/1dthermal.in ./pflotran_results/",stdin=None, stdout=None,stderr=None,shell=True)

kwargs2 = {"exeprg":exeprg,"mpirun":mpirun,"ncore":ncore,"niter":niter,"alpha":alpha,"da_time_win":da_time_win}
state_vector = Assimilator(nreaz,th1d,obs,**kwargs2)

subprocess.call("rm -f ./pflotran_results/1dthermal*.h5 ",stdin=None, stdout=None,stderr=None,shell=True)
subprocess.call("rm -f ./pflotran_results/1dthermal*.chk",stdin=None, stdout=None,stderr=None,shell=True)
subprocess.call("rm -f ./pflotran_results/1dthermal*.out",stdin=None, stdout=None,stderr=None,shell=True)
subprocess.call("rm -f ./pflotran_results/init_temp.h5",stdin=None, stdout=None,stderr=None,shell=True)

print("Done!")
