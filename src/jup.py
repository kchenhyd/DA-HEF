import ipywidgets as widgets
from ipywidgets import *
import numpy as np

def choose_environment():
    print("Please choose running environment:")
    envs = ['Local','Cluster']
    env = widgets.RadioButtons(
        value='Local',
        options=envs,
        rows=5,
        description='',
        disabled=False,
        style={'description_width': 'initial'},
#        layout=Layout(border='solid red')
    )
    display(env)
    return env

def run_job():
    print("Please choose the type of job running:")
    job_types = ['Jupyter notebook','Cluster']
    job_type = widgets.RadioButtons(
                        value = 'Cluster',
                        options=job_types,
                        rows=5,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
#        layout=Layout(border='solid red')
    )
    display(job_type)
    return job_type

def set_mpirun():
    print("Please enter the command for running parallel job:")
    mpirun = widgets.Text(
                        value = 'srun',
                        description='',
                        placeholder='e.g.,/home/user/petsc/gnu-c-debug/bin/mpirun',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(mpirun)
    return mpirun

def set_temp_resolution():
    print("Please enter resolution of temperature data to be assimilated (seconds):")
    temp_res = widgets.FloatText(
                        value=300.0,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(temp_res)
    return temp_res

def choose_error_type():
    print("Please choose the type of measurement error:")
    type_selected = widgets.RadioButtons(
                        value='relative',
                        options=['relative','absolute'],
                        rows=5,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'}
    )
    display(type_selected)
    return type_selected

def set_absolute_error():
    print("Please enter the absolute measurement error(C):")
    absolute_error = widgets.FloatText(
                        value=0.1,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(absolute_error)
    return absolute_error

def set_relative_error():
    print("Please enter the relative measurement error percentage:")
    relative_error = widgets.FloatText(
                        value=0.01,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(relative_error)
    return relative_error

def set_hydraulic_gradient_resolution():
    print("Please enter the hydraulic gradient observation resolution(sec):")
    grad_res = widgets.FloatText(
                        value=300.0,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(grad_res)
    return grad_res

def set_flow_direction_resolution():
    print("Please enter the flow direction observation resolution(sec):")
    dir_res = widgets.FloatText(
                        value=0,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(dir_res)
    return dir_res

def choose_thermistors(therm_locations):
    print('Choose thermistors (unit: m). Top three thermistors are selected by default. Press CTRL(PC) or Command(MAC) for multiple selection.  ')
    style = {'description_width': 'initial'}
    therm_locations_selected = widgets.SelectMultiple(
        value=tuple(np.array(therm_locations)[[0, 1, 4]]),
        options=therm_locations,
        rows=5,
        description='',
        disabled=False,
        style={'description_width': '200px'},
        layout=Layout(border='solid red')
    )
    display(therm_locations_selected)
    return therm_locations_selected

# check if the hydraulic gradient between the selected top and bottom thermistors are available
def has_hydraulic_gradient(therm_loc):
    if len(therm_loc)<3:
        raise Exception("Please choose at least three thermistors above.")
    print('Is hydraulic gradient between {} m and {} m available?\n'.format(max(therm_loc),min(therm_loc)))
    grad = widgets.RadioButtons(
        value='No',
        style={'description_width': '500px'},
        options=['Yes', 'No'],
        description='',
        disabled=False,
    )
    display(grad)
    return grad

def has_flow_direction():
    print("Is flow direction information available (use '-1' for upwelling and '1' for downwelling) ?\n")
    flow_dir = widgets.RadioButtons(
        value='No',
        options=['Yes', 'No'],
        description='',
        disabled=False
    )
    display(flow_dir)
    return flow_dir

def use_all_data():
    print('Do you want to estimate the flux in the entire time window?\n')
    all_data = widgets.RadioButtons(
                        value='Yes',
        style={'description_width': '500px'},
        options=['Yes', 'No'],
        description='',
        disabled=False
    )
    display(all_data)
    return all_data

def set_time_window_for_flux_estimation():
    time_window = []
    descriptions = ['Start time for flux estimation:','End time for flux estimation:']
    placeholders = ['e.g.,4/1/2017 0:00','e.g.,5/1/2017 0:00']
    print('Set the start and end time for flux estimation (format: "%m/%d/%Y, %H:%M", e.g. "4/1/2017 0:05"):\n')
    time_window.append(widgets.Text(
                    value='4/1/2017 0:00',
                    description=descriptions[0],
                    placeholder=placeholders[0],
                    disabled=False,
                    style={'description_width': 'initial'},
                    layout=Layout(border='solid red')
                    ))
    display(time_window[0])
    
    time_window.append(widgets.Text(
                    value='4/30/2017 23:40',
                    description=descriptions[1],
                    placeholder=placeholders[1],
                    disabled=False,
                    style={'description_width': 'initial'},
                    layout=Layout(border='solid red')
                    ))

    display(time_window[1])
    return time_window
    
    
def get_temperature_filepath():
    print("Path to the temperature data file. Default path is './data/':")
    filepath = widgets.Text(
                        value='./data/',
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(filepath)
    return filepath

def get_temperature_filename():
    print("Name of the temperature data file. Default name is 'temperature.csv':")
    filename = widgets.Text(
                        value='temperature.csv',
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(filename)
    return filename

def get_hydraulic_gradient_filepath():
    print("Path to the hydraulic gradient data file. The default path is the path you set earlier for temperature data:")
    filepath = widgets.Text(
                        value='./data/',
                        placeholder='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
                        )
    display(filepath)
    return filepath

def get_hydraulic_gradient_filename():
    print("File name of the hydraulic gradient data. The default path is the path you set earlier for temperature data:")
    filename = widgets.Text(
                        value='hydraulic head.csv',
                        placeholder='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
                        )
    display(filename)
    return filename

def get_flow_dir_filepath():
    print("Path to the flow direction data file. The default path is the path you set earlier for temperature data:")
    filepath = widgets.Text(
                        value='./data/',
                        description='',
                        placeholder='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
                        )
    display(filepath)
    return filepath

def get_flow_dir_filename():
    print("File name of the flow direction data. The default path is the path you set earlier for temperature data:")
    filename = widgets.Text(
                        value='flow direction.csv',
                        description='',
                        placeholder='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
                        )
    display(filename)
    return filename

def set_assimilation_timestep():
    print("Please enter the data assimilation time window(sec). The flux is considered to be constant in the timestep:")
    da_timestep = widgets.FloatText(
                        value = 900.0,
                        disabled= False,
                        layout=Layout(border='solid red')
    )
    display(da_timestep)
    return da_timestep

def set_prior_permeability():
    prior = []
    descriptions = ['Mean:','Standard Deviation:',
                   'Upper limit:','Lower limit:']
    default = ['-11.0','1.0','-9.0','-13.0']
    print('Set the prior permeability(log10(m^2)):\n')
    for i in range(4):
    	prior.append(widgets.FloatText(
                        value=default[i],
                        description=descriptions[i],
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
                        ))

    	display(prior[i])
    return prior   

def set_prior_flux():
    prior = []
    descriptions = ['Mean:','Standard Deviation:',
                   'Upper limit:','Lower limit:']
    default = ['0.0','0.5','5.0','-5.0']
    print('Set the prior flux (m/day):\n')
    for i in range(4):
    	prior.append(widgets.FloatText(
                        value=default[i],
                        description=descriptions[i],
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
                        ))

    	display(prior[i])
    return prior 

def set_prior_therm_cond():
    prior = []
    descriptions = ['Mean:','Standard Deviation:',
                   'Upper limit:','Lower limit:']
    default = ['1.5','0.5','2.5','0.5']
    print('Set the prior thermal conductivity (W/(mK)-1):\n')
    for i in range(4):
    	prior.append(widgets.FloatText(
                        value=default[i],
                        description=descriptions[i],
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
                        ))
    	display(prior[i])
    return prior 

def set_prior_porosity():
    prior = []
    descriptions = ['Mean:','Standard Deviation:',
                   'Upper limit:','Lower limit:']
    default = ['0.3','0.1','0.7','0.01']
    print('Set the prior porosity:\n')
    for i in range(4):
    	prior.append(widgets.FloatText(
                        value=default[i],
                        description=descriptions[i],
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
                        ))
    	display(prior[i])
    return prior 

def has_thermal_conductivity():
    print('Do you have thermal conductivity information?\n')
    has_therm = widgets.RadioButtons(
                        value = 'Yes',
                        style={'description_width': '500px'},
                        options=['Yes', 'No'],
                        description='',
                        disabled=False
    )
    display(has_therm)
    return has_therm

def set_thermal_conductivity():
    print("Please enter the thermal conductivity(W/(mK)-1):")
    therm_cond = widgets.FloatText(
                        value = 0.93,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(therm_cond)
    return therm_cond

def has_porosity():
    print('Do you have the porosity information?\n')
    has_porosity = widgets.RadioButtons(
                        value = 'Yes',
                        style={'description_width': '500px'},
                        options=['Yes', 'No'],
                        description='',
                        disabled=False
    )
    display(has_porosity)
    return has_porosity

def set_porosity():
    print("Please enter the porosity:")
    porosity = widgets.FloatText(
                        value = 0.43,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(porosity)
    return porosity

def set_nreaz():
    print("Please enter the number of realizations:")
    nreaz = widgets.IntText(
                        value=150,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(nreaz)
    return nreaz

def set_niter():
    print("Please enter the number of iterations:")
    niter = widgets.IntText(
                        value=4,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(niter)
    return niter

def set_iter_coeff(niter):
    print("Please enter the coefficient for each iteration:")
    coeff = []
    for i in range(int(niter)):
    	coeff.append(widgets.FloatText(
                        value='{}'.format(niter),
                        description='Iter.{}'.format(i+1),
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
                        ))
    	display(coeff[i])
    return coeff     

def set_exeprg():
    print("Please enter the path to the executable file of PFLOTRAN:")
    exeprg = widgets.Text(
                        value = '/global/project/projectdirs/pflotran/pflotran-cori-new/src/pflotran/pflotran',
                        description='',
                        placeholder='e.g.,~/pflotran/src/pflotran/pflotran',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(exeprg)
    return exeprg

def set_ncore():    
    print("Please enter the number of cores that could be used by PFLOTRAN:")
    ncore = widgets.IntText(
                        value = '150',
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(ncore)
    return ncore

def set_spinup():    
    print("Please enter the spinup time (days):")
    spinup = widgets.FloatText(
                        value=0.1,
                        description='',
                        disabled=False,
                        style={'description_width': 'initial'},
                        layout=Layout(border='solid red')
    )
    display(spinup)
    return spinup

def choose_flux_permeability():
    print('Do you want to estimate permeability using hydraulic gradient information?\n')
    esti_perm = widgets.RadioButtons(
                        value = 'Yes',
                        style={'description_width': '500px'},
                        options=['Yes', 'No'],
                        description='',
                        disabled=False
    )
    display(esti_perm)
    return esti_perm    

def replace_therm_cond(therm_cond):
    file = open('./pflotran_inputs/1dthermal.in','r+')
    lines = file.readlines()
    for i,line in enumerate(lines):
        if 'THERMAL_CONDUCTIVITY_WET' in line:
            lines[i] = '  THERMAL_CONDUCTIVITY_WET {}\n'.format(therm_cond)
            break
    file.close()
    file_new = open('./pflotran_inputs/1dthermal.in','w+')
    file_new.writelines(lines)
    file_new.close()
    return

def replace_porosity(porosity):
    file = open('./pflotran_inputs/1dthermal.in','r+')
    lines = file.readlines()
    for i,line in enumerate(lines):
        if 'POROSITY' in line:
            lines[i] = '  POROSITY {} \n'.format(porosity)
            break
    file.close()
    file_new = open('./pflotran_inputs/1dthermal.in','w+')
    file_new.writelines(lines)
    file_new.close()
    return

def save_parameter(fun_name,val):
    file = open('./src/jup.py','r+')
    lines = file.readlines()
    for idx, line in enumerate(lines):
        if fun_name == 'get_temperature_filepath' and fun_name in line:
            lines[idx+3] = ' '*24+"value='{}',\n".format(val)
            break
        if fun_name == 'get_temperature_filename' and fun_name in line:
            lines[idx+3] = ' '*24+"value='{}',\n".format(val)
            break
        if fun_name == 'choose_thermistors' and fun_name in line:
            lines[idx+4] = ' '*8+"value=tuple(np.array(therm_locations)[{}]),\n".format(val)
            break
        if fun_name == 'has_hydraulic_gradient' and fun_name in line:
            lines[idx+5] = ' '*8+"value='{}',\n".format(val)
            break
        if fun_name == 'get_hydraulic_gradient_filepath' and fun_name in line:
            lines[idx+3] = ' '*24+"value='{}',\n".format(val)
            break
        if fun_name == 'get_hydraulic_gradient_filename' and fun_name in line:
            lines[idx+3] = ' '*24+"value='{}',\n".format(val)
            break
        if fun_name == 'has_flow_direction' and fun_name in line:
            lines[idx+3] = ' '*8+"value='{}',\n".format(val)
            break
        if fun_name == 'get_flow_dir_filepath' and fun_name in line: 
            lines[idx+3] = ' '*24+"value='{}',\n".format(val)
            break
        if fun_name == 'get_flow_dir_filename' and fun_name in line: 
            lines[idx+3] = ' '*24+"value='{}',\n".format(val)
            break 
        if fun_name == 'use_all_data' and fun_name in line: 
            lines[idx+3] = ' '*24+"value='{}',\n".format(val)
            break        
        if fun_name == 'set_time_window_for_flux_estimation' and fun_name in line:
            lines[idx+6] = ' '*20+"value='{}',\n".format(val[0])
            lines[idx+16] = ' '*20+"value='{}',\n".format(val[1])
            break
        if fun_name == 'set_temp_resolution' and fun_name in line:
            lines[idx+3] = ' '*24+"value={},\n".format(val)
            break
        if fun_name == 'set_hydraulic_gradient_resolution' and fun_name in line:
            lines[idx+3] = ' '*24+"value={},\n".format(val)
            break
        if fun_name == 'choose_error_type' and fun_name in line:
            lines[idx+3] = ' '*24+"value='{}',\n".format(val)
            break
        if fun_name == 'set_absolute_error' and fun_name in line:
            lines[idx+3] = ' '*24+"value={},\n".format(val)
            break            
        if fun_name == 'set_relative_error' and fun_name in line:
            lines[idx+3] = ' '*24+"value={},\n".format(val)
            break 
        if fun_name == 'set_nreaz' and fun_name in line:
            lines[idx+3] = ' '*24+"value={},\n".format(val)
            break 
        if fun_name == 'set_niter' and fun_name in line:
            lines[idx+3] = ' '*24+"value={},\n".format(val)
            break             
        if fun_name == 'set_spinup' and fun_name in line:
            lines[idx+3] = ' '*24+"value={},\n".format(val)
            break
        if fun_name == 'set_prior_permeability' and fun_name in line:
            lines[idx+4] = ' '*4+"default = ['{}','{}','{}','{}']\n".format(val[0],val[1],val[2],val[3])
            break
        if fun_name == 'set_prior_flux' and fun_name in line:
            lines[idx+4] = ' '*4+"default = ['{}','{}','{}','{}']\n".format(val[0],val[1],val[2],val[3])
            break
        if fun_name == 'set_assimilation_timestep' and fun_name in line:
            lines[idx+3] = ' '*24+"value = {},\n".format(val)
            break
        if fun_name == 'choose_flux_permeability' and fun_name in line:
            lines[idx+3] = ' '*24+"value = '{}',\n".format(val)
            break
        if fun_name == 'has_thermal_conductivity' and fun_name in line:
            lines[idx+3] = ' '*24+"value = '{}',\n".format(val)
            break
        if fun_name == 'set_thermal_conductivity' and fun_name in line:
            lines[idx+3] = ' '*24+"value = {},\n".format(val)
            break
        if fun_name == 'set_prior_therm_cond' and fun_name in line:
            lines[idx+4] = ' '*4+"default = ['{}','{}','{}','{}']\n".format(val[0],val[1],val[2],val[3])
            break
        if fun_name == 'has_porosity' and fun_name in line:
            lines[idx+3] = ' '*24+"value = '{}',\n".format(val)
            break
        if fun_name == 'set_porosity' and fun_name in line:
            lines[idx+3] = ' '*24+"value = {},\n".format(val)
            break
        if fun_name == 'set_prior_porosity' and fun_name in line:
            lines[idx+4] = ' '*4+"default = ['{}','{}','{}','{}']\n".format(val[0],val[1],val[2],val[3])
            break
        if fun_name == 'set_exeprg' and fun_name in line:
            lines[idx+3] = ' '*24+"value = '{}',\n".format(val)
            break
        if fun_name == 'set_ncore' and fun_name in line:
            lines[idx+3] = ' '*24+"value = '{}',\n".format(val)
            break   
        if fun_name == 'set_mpirun' and fun_name in line:
            lines[idx+3] = ' '*24+"value = '{}',\n".format(val)
            break
        if fun_name == 'run_job' and fun_name in line:
            lines[idx+4] = ' '*24+"value = '{}',\n".format(val)
            break
    file.close()
    
    file_new = open('./src/jup.py','w+')
    file_new.writelines(lines)
    file_new.close()
    return                
                