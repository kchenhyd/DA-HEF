from src.pkg import *
from src.util import *


def Assimilator(nreaz,mod,obs,**kwargs):

################################################################################
#                  Ensemble Smoother-Multiple Data Assimilation
#
# reference: Emerick, A., and Reynolds, A.,2013, Ensemble smoother with Multiple
# data assimilation, Comput. Geosci.
#
# Paramters:
#    nreaz: number of realizations
#    infl: inflation coefficients
#    mod: TH1D object that contains all the model related parameters
#    obs: observation object that contains all the observation data
#    **kwargs:
#          "exeprg": path to the the executable program
#          "ncore_per_reaz": number of cores needed for running single realization
#          "niter": number of iterations for ensember smoother
#          "alpha": inflation coefficients for observation error covariance. The requirements
#                   for alpha is SUM(1/alpha(i)) = 1 where i is the number of iteration.
#          "da_time_win": time interval in which data are used for assimilation
#
# Return:
#        state_vector_out: updated state vector after every assimilation is appended to
#                          this vector
###############################################################################

    if 'permeability' in mod.da_para:
        with_head = True
    else:
        with_head = False

    if len(kwargs)!=6:
        raise Exception("Please provide the following arguments: exeprg, mpirun, ncore, niter, alpha, da_time_win")
    elif len(kwargs['alpha'])!=kwargs['niter']:
        raise Exception("length of alpha should be equal to number of iteration")
    else:
        exeprg = kwargs['exeprg']
        mpirun = kwargs['mpirun']
        ncore = kwargs['ncore']
        niter = kwargs['niter']
        alpha = kwargs['alpha']
        da_time_win = kwargs['da_time_win']

    if mod.spinup > 1e-10:
        run_spinup(nreaz,mod,obs,with_head,mpirun,exeprg,ncore)
        
    state_vector_out = copy.deepcopy(mod.state_vector)
    state_vector_iter = copy.deepcopy(mod.state_vector)
    state_vector_prior = copy.deepcopy(mod.state_vector)
    ibatch = 0 # one batch corresponds to one data assimilation time interval
    flux_change_win = []
    while ibatch<len(da_time_win):
        if not with_head and ibatch>0:
            state_vector_prior = np.concatenate((state_vector_prior,mod.state_vector),axis=0)
            
        print("Data assimilation time window (sec): {} ".format(da_time_win[ibatch]))
        idx_time_start = np.argmin(abs(obs.time-da_time_win[ibatch][0]))
        idx_time_end = np.argmin(abs(obs.time-da_time_win[ibatch][1]))
        obs_temp = copy.deepcopy(obs)
        obs_temp.time = obs_temp.time[idx_time_start:idx_time_end+1]
        obs_temp.value = obs_temp.value[idx_time_start:idx_time_end+1]
        obs_temp.ntime = len(obs_temp.time)
        for i in range(niter+1):
            if mod.spinup > 1e-10:
                if i == 0:
                    copy_chk()
                else:
                    load_chk()
                    
            if (mod.spinup < 1e-10) and (not with_head) and ibatch > 0:
                if i == 0:
                    copy_chk()
                else:
                    load_chk()              
                
            simu_ensemble = run_forward_simulation(nreaz,mod,obs_temp,with_head,exeprg,ncore,mpirun)
            if i == 0:
                simu_prior = simu_ensemble
                
            if with_head and i == 0:
                np.savetxt('./pflotran_results/simu_ensemble_out_with_head_prior.txt',simu_ensemble)
            if i == niter:
                continue
                
            if obs.error_type == 'absolute':
                obs_err_sd = obs_temp.error/3*np.ones(obs_temp.value[1:].shape)
            elif obs.error_type == 'relative':
                obs_err_sd = obs_temp.error/3*obs_temp.value[1:]
            obs_ensemble = np.repeat(obs_temp.value[1:].flatten('C').reshape((obs_temp.ntime-1)*obs_temp.nobs,1),nreaz,1)+np.sqrt(alpha[i])*np.dot(np.diag(obs_err_sd.flatten('C')),np.random.normal(0,1,nreaz*obs_temp.nobs*(obs_temp.ntime-1)).reshape(obs_temp.nobs*(obs_temp.ntime-1),nreaz))
            state_vector = mod.state_vector[:,:]
            cov_state_simu = np.cov(state_vector,simu_ensemble)[0:len(state_vector),len(state_vector):]
            cov_simu = np.cov(simu_ensemble)

            if obs.nobs*obs.ntime == 1:
                inv_cov_simu_obserr = np.array(1/(cov_simu+np.square(np.diag(obs_err_sd))))
            else:
                inv_cov_simu_obserr = la.inv(cov_simu+np.square(np.diag(obs_err_sd.flatten('C'))))
            kalman_gain = np.dot(cov_state_simu,inv_cov_simu_obserr)
            state_vector = state_vector+np.dot(kalman_gain,obs_ensemble-simu_ensemble)

            if with_head:
                print("Iteration {}:".format(i+1))
                print("  Mean of log(permeability) is (log(m^2)): {} ".format(np.mean(state_vector[0])))
                print("  STD of log(permeability) is (log(m^2)): {} \n".format(np.std(state_vector[0])))
            else:
                if i == niter-1:
                    print("  Mean of flux after {} iterations is (m/day): {} ".format(niter,np.mean(state_vector[0])))
                    print("  STD of flux after {} iterations is (m/day): {}  \n ".format(niter,np.std(state_vector[0])))

            for j in range(state_vector.shape[0]):
                state_vector[j,:][state_vector[j,:]<mod.state_vector_range[j,0]] = mod.state_vector_range[j,0]
                state_vector[j,:][state_vector[j,:]>mod.state_vector_range[j,1]] = mod.state_vector_range[j,1]
            mod.state_vector[:,:] = state_vector
            if with_head:
                state_vector_iter = np.concatenate((state_vector_iter,state_vector),axis=0)

        if not with_head:
            mod.state_vector[0,:] = np.random.normal(np.mean(state_vector[0]),0.5,nreaz)
            mod.state_vector[0,:][mod.state_vector[0,:]<mod.state_vector_range[0,0]] = mod.state_vector_range[0,0]
            mod.state_vector[0,:][mod.state_vector[0,:]>mod.state_vector_range[0,1]] = mod.state_vector_range[0,1]
            
            
        n = (obs_temp.ntime-1)*obs_temp.nobs
        if ibatch == 0:
            simu_ensemble_out = simu_ensemble
            simu_prior_out = simu_prior
            obs_out = np.concatenate((obs_temp.time.reshape(-1,1),obs_temp.value),axis=1)
            state_vector_out = state_vector
        else:
            simu_ensemble_out = np.concatenate((simu_ensemble_out,simu_ensemble),axis=0)
            simu_prior_out = np.concatenate((simu_prior_out,simu_prior),axis=0)
            obs_out = np.concatenate((obs_out,np.concatenate((obs_temp.time.reshape(-1,1),obs_temp.value),axis=1)),axis=0)      
            state_vector_out = np.concatenate((state_vector_out,state_vector),axis=0)
            
        if with_head:
            np.savetxt('./pflotran_results/state_vector_iter_with_head.txt',state_vector_iter)
            np.savetxt('./pflotran_results/simu_ensemble_out_with_head.txt',simu_ensemble_out)
            np.savetxt('./pflotran_results/obs_out_with_head.txt',obs_out)
        else:
            np.savetxt('./pflotran_results/state_vector_out_without_head.txt',state_vector_out)
            np.savetxt('./pflotran_results/simu_ensemble_out_without_head.txt',simu_ensemble_out)
            np.savetxt('./pflotran_results/simu_prior_out_without_head.txt',simu_prior_out)
            np.savetxt('./pflotran_results/obs_out_without_head.txt',obs_out)
            np.savetxt('./pflotran_results/state_vector_prior_without_head.txt',state_vector_prior)
        ibatch = ibatch+1
    np.savetxt('./pflotran_results/flux_change_win.txt',flux_change_win)
    return state_vector_out


