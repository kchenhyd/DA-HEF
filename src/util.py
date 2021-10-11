from src.pkg import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from datetime import timedelta

class TH1D:
###############################################################################
#
#                Define the model parameter class TH1D
#
# Details:
#
# TH1D.state_vector:Nd X Ne, where Nd is the number of varibles and Ne is the number
#               of realizations
# TH1D.state_vector_range: Nd X 2, the first column is the lower bound for each model
#               parameter and the second column is the upper bound for each paramter
# TH1D.z: 1-D array containing the coordinates of all grid centers
#
###############################################################################
    def __init__(self,da_para,nreaz,hz,spinup_length,**kwargs):

        self.da_para = da_para
        self.hz = hz
        self.dz = 0.01
        self.z =(np.arange(-hz,0,self.dz)+np.arange(-hz+self.dz,self.dz,self.dz))/2
        self.spinup = spinup_length

        if 'permeability' in da_para:
            logperm_mean = kwargs['logperm_mean']
            logperm_sd = kwargs['logperm_sd']
            logperm_low_bound = kwargs['logperm_low_bound']
            logperm_up_bound = kwargs['logperm_up_bound']
            init_logperm = np.random.normal(logperm_mean,logperm_sd,nreaz)
            logperm_range = np.array([logperm_low_bound,logperm_up_bound])
            init_logperm[init_logperm<logperm_range[0]] = logperm_range[0]
            init_logperm[init_logperm>logperm_range[1]] = logperm_range[1]
            init_perm = 10**(init_logperm)
        if 'thermal conductivity' in da_para:
            th_cond_mean = kwargs['th_cond_mean']
            th_cond_sd = kwargs['th_cond_sd']
            th_cond_low_bound = kwargs['th_cond_low_bound']
            th_cond_up_bound = kwargs['th_cond_up_bound']
            init_th_cond = np.random.normal(th_cond_mean,th_cond_sd,nreaz)
            th_cond_range = np.array([th_cond_low_bound,th_cond_up_bound])
            init_th_cond[init_th_cond<th_cond_range[0]] = th_cond_range[0]
            init_th_cond[init_th_cond>th_cond_range[1]] = th_cond_range[1]
        if 'porosity' in da_para:
            poro_mean = kwargs['poro_mean']
            poro_sd = kwargs['poro_sd']
            poro_low_bound = kwargs['poro_low_bound']
            poro_up_bound = kwargs['poro_up_bound']
            init_poro = np.random.normal(poro_mean,poro_sd,nreaz)
            poro_range = np.array([poro_low_bound,poro_up_bound])
            init_poro[init_poro<poro_range[0]] = poro_range[0]
            init_poro[init_poro>poro_range[1]] = poro_range[1]
        if 'flux' in da_para:
            flux_mean = kwargs['flux_mean']
            flux_sd = kwargs['flux_sd']
            flux_low_bound = kwargs['flux_low_bound']
            flux_up_bound = kwargs['flux_up_bound']
            flux_range = np.array([flux_low_bound,flux_up_bound])
            init_flux = np.random.normal(flux_mean,flux_sd,nreaz)
            init_flux[init_flux<flux_range[0]] = flux_range[0]
            init_flux[init_flux>flux_range[1]] = flux_range[1]

        if len(da_para) == 1:
            self.state_vector = np.zeros((1,nreaz))
            self.state_vector_range = np.zeros((1,2))
            if 'permeability' in da_para:
                self.state_vector = np.array([np.log10(init_perm)])
                self.state_vector_range = np.array([logperm_range])
            elif 'flux' in da_para:
                self.state_vector = np.array([init_flux])
                self.state_vector_range = np.array([flux_range])
            else:
                raise Exception("Please choose 'permeability' or 'flux'")
        elif len(da_para) == 2:
            self.state_vector = np.zeros((2,nreaz))
            if 'permeability' in da_para:
                self.state_vector = np.array([np.log10(init_perm)])
                self.state_vector_range = np.array([logperm_range])
            elif 'flux' in da_para:
                self.state_vector = np.array([init_flux])
                self.state_vector_range = np.array([flux_range])
            else:
                raise Exception("Please choose 'permeability' or 'flux'")
            if 'thermal conductivity' in da_para:
                self.state_vector = np.concatenate((self.state_vector,np.array([init_th_cond])))
                self.state_vector_range = np.concatenate((self.state_vector_range,np.array([th_cond_range])))
            elif 'porosity' in da_para:
                self.state_vector = np.concatenate((self.state_vector,np.array([init_poro])))
                self.state_vector_range = np.concatenate((self.state_vector_range,np.array([poro_range])))
            else:
                raise Exception("Please choose 'thermal conductivity' or 'porosity'")
        elif len(da_para) == 3:
            self.state_vector = np.zeros((3,nreaz))
            if 'permeability' in da_para:
                self.state_vector = np.array([np.log10(init_perm)])
                self.state_vector_range = np.array([logperm_range])
            elif 'flux' in da_para:
                self.state_vector = np.array([init_flux])
                self.state_vector_range = np.array([flux_range])
            else:
                raise Exception("Please choose 'permeability' or 'flux'")
            self.state_vector = np.concatenate((self.state_vector,np.array([init_th_cond])))
            self.state_vector = np.concatenate((self.state_vector,np.array([init_poro])))
            self.state_vector_range = np.concatenate((self.state_vector_range,np.array([th_cond_range])))
            self.state_vector_range = np.concatenate((self.state_vector_range,np.array([poro_range])))
        else:
            raise Exception("Maximum number of parameters is 3")


class Obs:
################################################################################
 #    obs: obsevation object which contains the observation time and dataself.
 #          obs.time: type: numpy.array, observation time series
 #          obs.ntime: total number of observation time points
 #          obs.value: type: numpy.array, observation data, e.g. Temperature
 #          obs.err_sd_ratio: ratio of the standard deviation of error to the observation value
 #          obs.coord: coordinates of the observation points
 #          obs.nobs: total number of observation points
################################################################################
    def __init__(self,obs_start_time,obs_end_time,obs_timestep,obs_coord,obs_error_type,obs_error,obs_data):
        self.start_time = obs_start_time
        self.end_time = obs_end_time
        self.timestep = obs_timestep
        self.time = obs_data[:,0]
        self.ntime = self.time.size
        self.value = obs_data[:,1:]
        self.coord = obs_coord
        self.error_type = obs_error_type
        self.error = obs_error
        self.nobs = obs_coord.size


def generate_dbase(nreaz,mod):
################################################################################
#
#   GenerateDbase: generate h5 file Dbase.h5 after each assimlation.
#   Dbase is a keyword in PFLOTRAN that makes the scalar value realization dependent.
#   In the TH1D model, the Dbase.h5 contains two datasets, the first is Permeability
#   and the second is ThermalConductivity. This function will be called in each iteration
#   to update the parameters.
#
#   Details:
#        nreaz: number of realizations
#        mod: model-specific object, for TH1D model, it contains the permeability
#            and thermal conductivity and associated hard limits
#
################################################################################
    filename = "./pflotran_results/Dbase.h5"
#    if os.path.isfile(filename):
#      h5file = h5py.File(filename,'r+')
#    else:
#      h5file = h5py.File(filename,'w')
    if os.path.isfile(filename):
        os.remove(filename)

    h5file = h5py.File(filename,'w')
    variables = []
    if 'permeability' in mod.da_para:
        variables.append("Permeability")
    elif 'flux' in mod.da_para:
        variables.append("Flux_top")
    else:
        raise Exception("Please choose 'permeability' or 'flux'")
    if 'thermal conductivity' in mod.da_para:
        variables.append('ThermalConductivity')
    if 'porosity' in mod.da_para:
        variables.append('Porosity')

    values = copy.deepcopy(mod.state_vector)
    if 'permeability' in mod.da_para:
        values[0] = 10**(values[0])
#     if 'flux' in mod.da_para:
#         values[0] = values[0]

    for i in range(len(variables)):
        if h5file.get(variables[i]):
            del h5file[variables[i]]
        h5dset = h5file.create_dataset(variables[i],data=values[i])
#    mod.state_vector[0] = np.log(mod.state_vector[0])
    h5file.close()

def generate_simulated_ensemble(nreaz,obs_coord,obs_time,z):
################################################################################
#
#   GenerateSimuEnsemble: generate the simulated data ensemble for assimlation.
#
#   Details:
#        nreaz: number of realizations
#        obs_coord: coordinates of observation points
#        obs_time: observation time
#        z: grid center of the one-dimensional model
#
################################################################################
    nobs = obs_coord.size
    obs_cell = np.zeros(nobs)
    ntime = obs_time.size
    for i in range(nobs):
        obs_cell[i] = np.argmin(np.absolute(z-obs_coord[i]))+1
    obs_cell = obs_cell.astype(int)
    simu_ensemble = np.zeros((nobs*(ntime-1),nreaz))
    for ireaz in range(nreaz):
        obs_temp = np.zeros(nobs*(ntime-1))
        j = 0
        for itime in obs_time[1:]:
            h5f = h5py.File("./pflotran_results/1dthermalR{}.h5".format(ireaz+1),'r')
            group_time = "Time:"+str(" %12.5E" % itime)+" s"
            dataset_temp = "Temperature [C]"
            obs_temp[j*nobs:(j+1)*nobs] = h5f[group_time][dataset_temp][0][0][obs_cell]
            j = j+1
            h5f.close()
        simu_ensemble[:,ireaz] = obs_temp

    return simu_ensemble

def run_forward_simulation(nreaz,mod,obs,with_head,exeprg,ncore,mpirun):
    if with_head:
        FNULL = open(os.devnull,'w')
        generate_dbase(nreaz,mod)
        make_pflotran_input(mod,obs,with_head,spinup=False)
        subprocess.call("./src/pflotran.sh {} {} {} {} ".format(nreaz,ncore,mpirun,exeprg),stdin=None,stdout=FNULL,stderr=None,shell=True)

        simu_ensemble = generate_simulated_ensemble(nreaz,obs.coord,obs.time,mod.z)
    else:
        FNULL = open(os.devnull,'w')
        generate_dbase(nreaz,mod)
        make_pflotran_input(mod,obs,with_head,spinup=False)
        subprocess.call(["./src/pflotran.sh {} {} {} {} ".format(nreaz,ncore,mpirun,exeprg)],stdin=None,stdout=FNULL,stderr=None,shell=True)

        simu_ensemble = generate_simulated_ensemble(nreaz,obs.coord,obs.time,mod.z)

    return simu_ensemble

def run_spinup(nreaz,mod,obs,with_head,mpirun,exeprg,ncore):
    print("Spinup starts...")
    if with_head:
        FNULL = open(os.devnull,'w')
        generate_dbase(nreaz,mod)
        make_pflotran_input(mod,obs,with_head,spinup=True)
        subprocess.call("./src/pflotran.sh {} {} {} {} ".format(nreaz,ncore,mpirun,exeprg),stdin=None,stdout=FNULL,stderr=None,shell=True)
    else:
        FNULL = open(os.devnull,'w')
        generate_dbase(nreaz,mod)
        make_pflotran_input(mod,obs,with_head,spinup=True)
        subprocess.call("./src/pflotran.sh {} {} {} {} ".format(nreaz,ncore,mpirun,exeprg),stdin=None,stdout=FNULL,stderr=None,shell=True)
    print("Spinup ends. \n")


def initialize_restart_file(nreaz,obs_time):
    for ireaz in range(nreaz):
        h5f = h5py.File("./pflotran_results/1dthermalR{}.h5".format(ireaz+1),'r+')
        for itime in obs_time[1:]:
            group_time = "Time:"+str(" %12.5E" % itime)+" s"
            del h5f[group_time]
        h5f.close()

def copy_chk():
    subprocess.call("cp ./pflotran_results/1dthermal*restart.chk ./pflotran_results/chk_temp",stdin=None, stdout=None,stderr=None,shell=True)

def load_chk():
    subprocess.call("cp ./pflotran_results/chk_temp/1dthermal*restart.chk ./pflotran_results/",stdin=None, stdout=None,stderr=None,shell=True)

def make_pflotran_input(mod,obs,with_head,spinup):
#    obs_win = np.loadtxt('./observation/da_time_win.dat')
    with open('./pflotran_results/1dthermal.in','r+') as f:
        pflotranin = f.readlines()
        if with_head:
            for i,s in enumerate(pflotranin):
                if "PERM_ISO" in s:
                    pflotranin[i] = "    PERM_ISO DBASE_VALUE Permeability" + "\n"
#                 if 'TEMPERATURE FILE ../pflotran_inputs/temp_top.dat' in s:
#                     pflotranin[i-3] = "  DATUM FILE ../pflotran_inputs/head_top.dat" + "\n"
#                 if 'TEMPERATURE FILE ../pflotran_inputs/temp_bottom.dat' in s:
#                     pflotranin[i-2] = "  DATUM FILE ../pflotran_inputs/head_bottom.dat" + "\n"
                if 'NXYZ' in s:
                    pflotranin[i] = "  NXYZ 1 1 {}".format(int(mod.hz*100)) + "\n"
                    pflotranin[i+2] = "    0.d0 0.d0 {}".format(-mod.hz) + "d0" + "\n"
                if 'REGION all' in s and 'COORDINATES' in pflotranin[i+1]:
                    pflotranin[i+2] = "    0.d0 0.d0 {}".format(-mod.hz) + "d0" + "\n"
                if 'REGION bottom' in s and "FACE" in pflotranin[i+1]:
                    pflotranin[i+3] = "    0.d0 0.d0 {}".format(-mod.hz) + "d0" + "\n"
                    pflotranin[i+4] = "    1.d0 1.d0 {}".format(-mod.hz) + "d0" + "\n"
                if 'SNAPSHOT_FILE' in s:
                    if spinup:
                        pflotranin[i+1] = "   PERIODIC TIME {}".format(mod.spinup*86400) + " s" +"\n"
                    else:
                        pflotranin[i+1] = "   PERIODIC TIME {}".format(obs.timestep) + " s" +"\n"
                if 'FLOW_CONDITION flow_top' in s and "TYPE" in pflotranin[i+1]:
                    pflotranin[i+5] = "  DATUM FILE ../pflotran_inputs/head_top.dat" + "\n"
                if 'FLOW_CONDITION flow_bottom' in s and "TYPE" in pflotranin[i+1]:
                    pflotranin[i+5] = "  DATUM FILE ../pflotran_inputs/head_bottom.dat" + "\n"
#                if 'FLOW_CONDITION initial' in s and "TYPE" in pflotranin[i+1]:
#                    pflotranin[i+6] = "  TEMPERATURE " + str(np.mean(obs.value[0,:])) + "d0" + "\n"
                if 'THERMAL_CONDUCTIVITY_WET' in s:
                    if 'thermal conductivity' in mod.da_para:
                        pflotranin[i] = "  THERMAL_CONDUCTIVITY_WET DBASE_VALUE ThermalConductivity" + "\n"
                if 'POROSITY' in s:
                    if 'porosity' in mod.da_para:
                        pflotranin[i] = "  POROSITY DBASE_VALUE Porosity" + "\n"
                if 'MODE TH' in s:
                    if not spinup:
                        pflotranin[i+1] = "      OPTIONS"+"\n"
                        pflotranin[i+2] = "        REVERT_PARAMETERS_ON_RESTART"+"\n"
                        pflotranin[i+3] = "      /"+"\n"
                if "FILENAME 1dthermal" in s:
                    if (not spinup) and (obs.time[0] > 1e-10):
                         pflotranin[i-1] = "  RESTART"+"\n"
                         pflotranin[i] = "    FILENAME 1dthermal-restart.chk "+" \n"
                         pflotranin[i+1] = "    REALIZATION_DEPENDENT"+"\n"
                         pflotranin[i+2] = "#    RESET_TO_TIME_ZERO /" +"\n"
                if 'FINAL_TIME' in s:
                    if spinup:
                        pflotranin[i] = "  FINAL_TIME " + str(mod.spinup*86400) + "  sec" + "\n"
                    else:
                        pflotranin[i] = "  FINAL_TIME " + str(obs.time[-1]) + "  sec" + "\n"

        else:
            for i,s in enumerate(pflotranin):
                if 'NXYZ' in s:
                    pflotranin[i] = "  NXYZ 1 1 {}".format(int(mod.hz*100)) + "\n"
                    pflotranin[i+2] = "    0.d0 0.d0 {}".format(-mod.hz) + "d0" + "\n"
                if 'REGION all' in s and 'COORDINATES' in pflotranin[i+1]:
                    pflotranin[i+2] = "    0.d0 0.d0 {}".format(-mod.hz) + "d0" + "\n"
                if 'REGION bottom' in s and "FACE" in pflotranin[i+1]:
                    pflotranin[i+3] = "    0.d0 0.d0 {}".format(-mod.hz) + "d0" + "\n"
                    pflotranin[i+4] = "    1.d0 1.d0 {}".format(-mod.hz) + "d0" + "\n"
                if 'SNAPSHOT_FILE' in s:
                    if spinup:
                        pflotranin[i+1] = "   PERIODIC TIME {}".format(259200) + " s" +"\n"
                    else:
                        pflotranin[i+1] = "    PERIODIC TIME {}".format(obs.timestep) + " s" +"\n"
                if "FLOW_CONDITION flow_top" in s and "TYPE" in pflotranin[i+1]:
                    pflotranin[i+2] = "    FLUX NEUMANN" + "\n"
                    pflotranin[i+5] = "\n"
                    pflotranin[i+6] = "  FLUX DBASE_VALUE Flux_top m/day" +"\n"
#                if 'FLOW_CONDITION initial' in s and "TYPE" in pflotranin[i+1]:
#                    pflotranin[i+6] = "  TEMPERATURE " + str(np.mean(obs.value[0,:])) + "d0" + "\n"
                if 'THERMAL_CONDUCTIVITY_WET' in s:
                    if 'thermal conductivity' in mod.da_para:
                        pflotranin[i] = "  THERMAL_CONDUCTIVITY_WET DBASE_VALUE ThermalConductivity" + "\n"
                if 'POROSITY' in s:
                    if 'porosity' in mod.da_para:
                        pflotranin[i] = "  POROSITY DBASE_VALUE Porosity" + "\n"
                if "FINAL_TIME" in s:
                    pflotranin[i] = "  FINAL_TIME {} sec".format(obs.time[-1])+"\n"
                if "MODE TH" in s:
                    if not spinup:
                     pflotranin[i+1] = "      OPTIONS"+"\n"
                     pflotranin[i+2] = "        REVERT_PARAMETERS_ON_RESTART"+"\n"
                     pflotranin[i+3] = "      /"+"\n"
                if "FILENAME 1dthermal" in s:
                    if (not spinup) and (obs.time[0] > 1e-10):
#                        print(pflotranin[i+2])
                        pflotranin[i-1] = "  RESTART"+"\n"
                        pflotranin[i] = "    FILENAME 1dthermal-restart.chk "+" \n"
                        pflotranin[i+1] = "    REALIZATION_DEPENDENT"+"\n"
#                        if obs.time[0] < 2e4:
#                          pflotranin[i+2] = "#    RESET_TO_TIME_ZERO /" +"\n"
#                        else:
#                          pflotranin[i+2] = "#    RESET_TO_TIME_ZERO /" +"\n"
                if "FINAL_TIME" in s:
                    if spinup:
                        pflotranin[i] = "  FINAL_TIME {} sec".format(mod.spinup*86400)+"\n"
                    else:
                        pflotranin[i] = "  FINAL_TIME {} sec".format(obs.time[-1])+"\n"
    f.close()

    os.remove('./pflotran_results/1dthermal.in')
    with open('./pflotran_results/1dthermal.in','w') as new_f:
        new_f.writelines(pflotranin)
    new_f.close()
    return

def draw_plot(data, edge_color, fill_color1,fill_color2):
    bp = plt.boxplot(data, patch_artist=True, showfliers=False)
    colors = [fill_color1,fill_color2,fill_color2,fill_color2,fill_color2]
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch,color in zip(bp['boxes'],colors):
        patch.set(facecolor=color)

def plot_para_without_head(da_para,da_timestep,obs_timestep,nreaz):
    state_vector = np.loadtxt('./pflotran_results/state_vector_out_without_head.txt')
    obs_data = np.loadtxt('./pflotran_results/obs_out_without_head.txt')


    sec_to_day = 3600*24
    obs_data[:,0] = obs_data[:,0]/sec_to_day

    with open('./pflotran_inputs/1dthermal.in','r+') as f:
        pflotranin = f.readlines()
        for i,s in enumerate(pflotranin):
            if 'PERM_ISO' in s:
                perm = float(s.split()[1])
    f.close()



    fig = plt.figure(num=1,dpi=300)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    plt.tight_layout()

    state_vector = state_vector[len(da_para):,:]
    for i in range(len(da_para)):
        plt.subplot(len(da_para),1,i+1)
        if i == 0:
            m = int(da_timestep/obs_timestep)

            n = int(state_vector.shape[0]/len(da_para))
            obs_time = np.arange(0,n*da_timestep,da_timestep)/sec_to_day
            line1, = plt.plot(obs_time,state_vector[:,0],'chocolate',linewidth=0.3)
            for j in range(nreaz-1):
                plt.plot(obs_time,state_vector[:,j+1],'chocolate',linewidth=0.3)
            line2, = plt.plot(obs_time,np.mean(state_vector,axis=1),'b',linewidth=1)
            line3, = plt.plot(obs_time,np.percentile(state_vector,95,axis=1),'b--',linewidth=0.5)
            line4, = plt.plot(obs_time,np.percentile(state_vector,5,axis=1),'b--',linewidth=0.5)

            if len(da_para) == 1:
                plt.xlabel('Time (day)')
            plt.ylabel('q (m/d)')
            ymax = np.ceil(max(np.mean(hef_vector,axis=1)))
            ymin = np.floor(min(np.mean(hef_vector,axis=1)))
            nticks = 4
            plt.yticks(np.arange(ymin, ymax+0.1, (ymax-ymin)/nticks))
            plt.legend((line1,line2,line3,line4),('Posterior','Mean','95%','5%'),frameon=False,ncol=4)
        else:
            plt.subplot(len(da_para),1,i+1)
            line1, = plt.plot(obs_time,state_vector[i::len(da_para),0],'chocolate',linewidth=0.3)
            for j in range(nreaz-1):
                plt.plot(obs_time,state_vector[i::len(da_para),j+1],'chocolate',linewidth=0.3)
            line2, = plt.plot(obs_time,np.mean(state_vector[i::len(da_para),:],axis=1),'b',linewidth=1)
            line3, = plt.plot(obs_time,np.percentile(state_vector[i::len(da_para),:],95,axis=1),'b--',linewidth=0.5)
            line4, = plt.plot(obs_time,np.percentile(state_vector[i::len(da_para),:],5,axis=1),'b--',linewidth=0.5)

            if da_para[i] == 'thermal conductivity':
                plt.ylabel('$\lambda$ (W/m$\cdot$K)')
            else:
                plt.ylabel('$\phi$')
            ymax = np.ceil(max(np.mean(state_vector[i::len(da_para),:],axis=1)))
            ymin = np.floor(min(np.mean(state_vector[i::len(da_para),:],axis=1)))
            nticks = 4
            plt.yticks(np.arange(ymin, ymax+0.1, (ymax-ymin)/nticks))

            if i == len(da_para)-1:
                plt.xlabel('Time (day)')

def plot_para_with_head(da_para):
    state_vector = np.loadtxt('./pflotran_results/state_vector_out_with_head.txt')
    obs_data = np.loadtxt('./pflotran_results/obs_out_with_head.txt')

    sec_to_day = 3600*24
    obs_data[:,0] = obs_data[:,0]/sec_to_day

    fig = plt.figure(num=1,dpi=300)
    fig.subplots_adjust(wspace=0.3)
    plt.tight_layout()

    for i in range(len(da_para)):
        plt.subplot(1,len(da_para),i+1)
        perm = np.array([state_vector[i,:],state_vector[i+len(da_para),:]])
        draw_plot(np.transpose(perm),'red','cyan','tan')
        if da_para[i] == 'permeability':
            plt.ylabel('log$_{10}(k$) (m$^{2}$)')
        elif da_para[i] == 'thermal conductivity':
            plt.ylabel('Thermal Cond. (W/m$\cdot$K)')
        else:
            plt.ylabel('Porosity')

        plt.title('{}'.format(da_para[i]))
        plt.xticks([1,2],['Prior','Posterior'])

def plot_hef_with_gradient(da_para,hz,obs_start_time,obs_end_time,init_datetime,state_file,obs_file,head_top_file,head_bot_file):
    state_vector = np.loadtxt(state_file)
    obs_data = np.loadtxt(obs_file)
    head_top = np.loadtxt(head_top_file)[-obs_data.shape[0]:,:]
    head_bottom = np.loadtxt(head_bot_file)[-obs_data.shape[0]:,:]
    head_top = head_top[:,:]
    head_bottom = head_bottom[:,:]
    nreaz = state_vector.shape[1]

    for i in range(int(state_vector.shape[0]/len(da_para))):
        if i == 0:
            print("Prior permeability:")
            print("  Mean of log(permeability) is (log(m^2)): {} ".format(np.mean(state_vector[i])))
            print("  STD of log(permeability) is (log(m^2)): {} \n".format(np.std(state_vector[i])))
        else:
            print("Iteration {}:".format(i))
            print("  Mean of log(permeability) is (log(m^2)): {} ".format(np.mean(state_vector[i*len(da_para)])))
            print("  STD of log(permeability) is (log(m^2)): {} \n".format(np.std(state_vector[i*len(da_para)])))
    init_datetime = datetime.strptime(init_datetime,"%m/%d/%Y %H:%M")
    sec_to_day = 3600*24
    perm = 10**(state_vector[-len(da_para),:])
    temp = obs_data[1:,1]+273.15
    hy_grad = (head_top[1:,3]-head_bottom[1:,3])/hz
    viscosity = 1e-6*(280.68*(temp/300)**(-1.9)+511.45*(temp/300)**(-7.7)+61.131*(temp/300)**(-19.6)+0.45903*(temp/300)**(-40))
    hef_vector = sec_to_day*1000*9.8*np.matmul(np.multiply(np.transpose(hy_grad),1/viscosity).reshape(-1,1),perm.reshape(1,-1))
    fig = plt.figure(num=1,dpi=150)

    n = int(state_vector.shape[0]/len(da_para))
    obs_time = obs_data[1:,0]/sec_to_day
    obs_time1 = []
    for i in range(obs_time.shape[0]):
        obs_time1.append(init_datetime+timedelta(seconds=obs_data[i,0]))

    line1, = plt.plot(obs_time1,hef_vector[:,0],'chocolate',linewidth=0.3)
    for j in range(nreaz-1):
        plt.plot(obs_time1,hef_vector[:,j+1],'chocolate',linewidth=0.3)
    line2, = plt.plot(obs_time1,np.mean(hef_vector,axis=1),'b',linewidth=1)
    line3, = plt.plot(obs_time1,np.percentile(hef_vector,95,axis=1),'b--',linewidth=0.5)
    line4, = plt.plot(obs_time1,np.percentile(hef_vector,5,axis=1),'b--',linewidth=0.5)
    line5, = plt.plot(obs_time1,np.zeros(len(obs_time)),'k--',linewidth=0.3)

    plt.xlabel('Time (day)')
    plt.ylabel('HEF (m/d)')
    plt.legend((line1,line2,line3,line4),('Posterior','Mean','95%','5%'),frameon=False)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.savefig('./pflotran_results/hef_with_head.png',dpi=150)

def plot_hef_without_gradient(da_para,da_timestep,obs_timestep,init_datetime,state_file):
    state_vector = np.loadtxt(state_file)

    nreaz = state_vector.shape[1]
    init_datetime = datetime.strptime(init_datetime,"%m/%d/%Y %H:%M")
    state_vector = state_vector[len(da_para):,:]
    m = int(da_timestep/obs_timestep)
    hef_vector = state_vector[::len(da_para),:]

    fig = plt.figure(num=1,dpi=150)

    n = int(state_vector.shape[0]/len(da_para))
    obs_time = np.arange(0,n*da_timestep,da_timestep)
    obs_time1 = []
    datetime_temp = init_datetime
    for i in range(obs_time.shape[0]):
        datetime_temp = datetime_temp+timedelta(seconds=da_timestep)
        obs_time1.append(datetime_temp)
    num = min(len(obs_time1),hef_vector.shape[0])
    line1, = plt.plot(obs_time1[:num],hef_vector[:num,0],'chocolate',linewidth=0.3)
    for j in range(nreaz-1):
        plt.plot(obs_time1[:num],hef_vector[:num,j+1],'chocolate',linewidth=0.3)
    line2, = plt.plot(obs_time1[:num],np.mean(hef_vector[:num,:],axis=1),'b',linewidth=1)
    line3, = plt.plot(obs_time1[:num],np.percentile(hef_vector[:num,:],95,axis=1),'b--',linewidth=0.5)
    line4, = plt.plot(obs_time1[:num],np.percentile(hef_vector[:num,:],5,axis=1),'b--',linewidth=0.5)
    line5, = plt.plot(obs_time1[:num],np.zeros(len(obs_time)),'k--',linewidth=0.3)
    plt.xlabel('Time (day)')
    plt.ylabel('HEF (m/d)')
    plt.legend((line1,line2,line3,line4),('Posterior','Mean','95%','5%'),frameon=False)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.savefig('./pflotran_results/hef_without_head.png',dpi=150)

def plot_temp_with_gradient(obs_coord,init_datetime,state_file,pred_file,obs_file,plot_type):
    state_vector = np.loadtxt(state_file)
    simu_ensemble = np.loadtxt(pred_file)
    obs_data = np.loadtxt(obs_file)

    nreaz = state_vector.shape[1]
    init_datetime = datetime.strptime(init_datetime,"%m/%d/%Y %H:%M")
    sec_to_day = 3600*24
    obs_time = obs_data[:,0]/sec_to_day
    obs_time1 = []
    for i in range(obs_time.shape[0]):
        obs_time1.append(init_datetime+timedelta(seconds=obs_data[i,0]))

    fig = plt.figure(num=1,dpi=150)
    nobs = len(obs_coord)

    for i in range(nobs):
        line1, = plt.plot(obs_time1,simu_ensemble[i::nobs,0],'chocolate',linewidth=0.3)
        for j in range(nreaz-1):
            plt.plot(obs_time1,simu_ensemble[i::nobs,j+1],'chocolate',linewidth=0.3)
        line2, = plt.plot(obs_time1,np.mean(simu_ensemble[i::nobs,:],axis=1),'b',linewidth=1)
        line3, = plt.plot(obs_time1,obs_data[:,i+1],'k',linewidth=1)

        plt.xlabel('Time (day)')
        plt.ylabel('Temperature ($^\circ$C)')
        plt.legend((line1,line2,line3),('Posterior','Mean','Observation'),frameon=False)
        plt.title('{}: Obs. point {} m'.format(plot_type,obs_coord[i]))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()

def plot_temp_without_gradient(therm_loc,da_para,obs_timestep,init_datetime,state_file,pred_file,obs_file,plot_type):
    state_vector = np.loadtxt(state_file)
    simu_ensemble = np.loadtxt(pred_file)
    obs_data = np.loadtxt(obs_file)

    nreaz = state_vector.shape[1]
    nobs = len(therm_loc)-2
    init_datetime = datetime.strptime(init_datetime,"%m/%d/%Y %H:%M")

    sec_to_day = 3600*24
    obs_time = obs_data[:,0]/sec_to_day
    obs_time1 = []
    for i in range(obs_data.shape[0]):
        obs_time1.append(init_datetime+timedelta(seconds=obs_data[i,0]))
    fig = plt.figure(num=1,dpi=150)
    for i in range(nobs):
        plt.subplot(nobs,1,i+1)
        line1, = plt.plot(obs_time1,obs_data[:,i+1],'k',linewidth=1)
        line2, = plt.plot(obs_time1,simu_ensemble[i::nobs,0],'chocolate',linewidth=0.3)
        for j in range(nreaz-1):
            plt.plot(obs_time1,simu_ensemble[i::nobs,j+1],'chocolate',linewidth=0.3)
        line3, = plt.plot(obs_time1,np.mean(simu_ensemble[i::nobs,:],axis=1),'b',linewidth=1)


        plt.xlabel('Time (day)')
        plt.ylabel('Temperature ($^\circ$C)')
        plt.legend((line1,line2,line3),('Observation','Posterior','Mean'),frameon=False)
        plt.title('{}: Obs. point {} m'.format(plot_type,therm_loc[i+1]))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()

def plot_temp_data(therm_loc,spinup_length,init_datetime):
    temp_top = np.loadtxt('./pflotran_inputs/temp_top.dat',skiprows=1)
    temp_bot = np.loadtxt('./pflotran_inputs/temp_bottom.dat',skiprows=1)
    temp_obs = np.loadtxt('./observation/obs_data.dat',skiprows=1)

    init_datetime = datetime.strptime(init_datetime,"%m/%d/%Y %H:%M")
    day_to_second = 86400
    start_datetime = init_datetime+timedelta(days=spinup_length)
    start_idx = np.argmin(np.abs(temp_obs[:,0]-spinup_length*day_to_second))
    obs_coord = np.array(therm_loc[1:-1])
    time = []
    for i in range(temp_obs[start_idx:,:].shape[0]):
        time.append(start_datetime+timedelta(seconds=temp_obs[i,0]))

    fig = plt.figure(num=1,dpi=150)

    line1, = plt.plot(time,temp_top[start_idx:,1],'b',linewidth=1)
    line2, = plt.plot(time,temp_bot[start_idx:,1],'r',linewidth=1)

    lines = []
    legends = []
    colors = ['g','c','k','y']
    for i in range(len(obs_coord)):
        line, = plt.plot(time,temp_obs[start_idx:,i+1],linewidth=1,color=colors[i])
        lines.append(line)
        legends.append(str(obs_coord[i])+' m')

    plt.xlabel('Time (day)')
    plt.ylabel('Temperature ($^\circ$C)')
    plt.legend([line1]+lines+[line2],['{} m'.format(therm_loc[0])]+legends+['{} m'.format(therm_loc[-1])],frameon=False)
    plt.title('Temperature data')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.savefig('./pflotran_results/temperature_data.png',dpi=150)
