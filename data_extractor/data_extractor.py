import os,sys,subprocess,glob,cftime,importlib,pickle,itertools
from datetime import datetime
import xarray as xr
import numpy as np
sys.path.append('../')

from ensembles.ensemble_GKLT import ensemble_GKLT


realm_dict = {
    'atm' : 'atmos',
    'lnd' : 'land',
}

variable_dict = {
    'TREFHT' : 'tas',
    'U200' : 'ua200',
    'U500' : 'ua500',
    'V500' : 'va500',
    'Z500' : 'zg500',
    'PSL' : 'psl',
    'SST' : 'tos',
    'SOILWATER_10CM' : 'mrsos',
    'NEP' : 'nep',
    'pr' : 'pr',
}

def extract_rea(
    exp,
    variable = 'Z500',
    realm = 'atm',
    h_identifier = 'h1',
    time_frequency = 'day',
    preprocessing_module = None,
    overwrite = False,
):

    if preprocessing_module is not None:
        preprocessor = preprocessing_module.preprocessor
        name_addition = preprocessing_module.name_addition
        preprocessing_attr = preprocessing_module.preprocessing_attr
    else:
        preprocessor = None
        name_addition = ''
        preprocessing_attr = 'no preprocessing'   

    ens = ensemble_GKLT(exp)
    
    naming_d = {
        "project": 'REA_output',
        "product": exp.product_name,
        "institute": 'NCAR',
        "model": 'CESM2',
        "experiment": f"{exp.initial_conditions_name}-x{exp.experiment_identifier[1]}",
        "time_frequency": time_frequency,
        "realm": realm_dict[realm],
        "variable": variable_dict[variable]+name_addition,
    }

    if variable in ['SST']:
        naming_d['realm'] = 'ocean'

    trajectory_names = sorted([s for s in ens._sim_names if len(s.split('/')) == ens._exp.n_steps])
    trajs = []
    for i,sim_name in enumerate(trajectory_names):
        ens_name = f"ens{str(i+1).zfill(3)}"
        out_dir = '/'.join([exp.dir_work] + [v for k,v in naming_d.items()] + [ens_name])

    
        out_file_name = f"{out_dir}/{variable_dict[variable]+name_addition}_{time_frequency}_CESM2_{naming_d['experiment']}_{ens_name}_{exp.initial_condition_fake_year}.nc"
        print(out_file_name)
        if os.path.isfile(out_file_name) == False or overwrite:
            todos = [['/'.join(sim_name.split('/')[:step])] for step in range(1,exp.n_steps+1)]
            l = []
            for step in range(1,exp.n_steps+1):
                _sim_name_of_step_ = '/'.join(sim_name.split('/')[:step])
                try:
                    h_file = glob.glob(f"{exp.dir_archive_post}/{_sim_name_of_step_}/{realm}/hist/*{h_identifier}*.nc")[0]
                    with xr.open_mfdataset(h_file, preprocess=preprocessor) as nc:
                        l.append(nc[variable])
                except:
                    #print(f'fail {h_file}')
                    pass
            if len(l) == exp.n_steps:
                x = xr.merge(l)[variable].sortby('time')
                x = x.assign_coords(time=xr.date_range(f"{exp.initial_condition_fake_year}-{exp.start_date_in_year}", periods=exp.n_days*exp.n_steps + 1)[1:])
                if time_frequency == 'mon':
                    x = x.resample(time='ME').mean()
                x = x.assign_coords(sim=sim_name)
                ds = xr.Dataset({variable_dict[variable]:x})
                ds.attrs = nc.attrs
                ds.attrs['simulation_name'] = sim_name
                ds.attrs['initial_condition'] = exp.initial_conditions[i]
                ds.attrs['initial_condition_year'] = exp.initial_conditions[i].split('/')[-1].split('-')[0]
                ds.attrs['compset'] = exp.compset
                ds.attrs['readme'] = exp.git_repo
                ds.attrs['preprocessing'] = preprocessing_attr
                ds['time'].attrs['comment'] = f'This simulation represents the climate state of {exp.initial_conditions_name}. The year in the time axis can be ignored.'

                os.makedirs(out_dir, exist_ok=True)
                ds.to_netcdf(out_file_name)
            #else:
            #    print('fail', l)


def extract_initial(
    exp,
    variable = 'Z500',
    realm = 'atm',
    h_identifier = 'h1',
    time_frequency = 'day',
    preprocessing_module = None,
    overwrite = False,
):

    if preprocessing_module is not None:
        preprocessor = preprocessing_module.preprocessor
        name_addition = preprocessing_module.name_addition
        preprocessing_attr = preprocessing_module.preprocessing_attr
    else:
        preprocessor = None
        name_addition = ''
        preprocessing_attr = 'no preprocessing'        

    naming_d = {
        "project": 'REA_output',
        "product": exp.product_name,
        "institute": 'NCAR',
        "model": 'CESM2',
        "experiment": f"{exp.initial_conditions_name}-initial",
        "time_frequency": time_frequency,
        "realm": realm_dict[realm],
        "variable": variable_dict[variable]+name_addition,
    }

    if variable in ['SST']:
        naming_d['realm'] = 'ocean'

    for i,initial_condition in enumerate(exp.initial_conditions):
        ens_name = f"ens{str(i+1).zfill(3)}"
        out_dir = '/'.join([exp.dir_work] + [v for k,v in naming_d.items()] + [ens_name])
        case_identifier = initial_condition.split('.')[-4] + '_' + initial_condition.split('/')[-1].split('-')[0]
        out_file_name = f"{out_dir}/{variable_dict[variable]+name_addition}_{time_frequency}_CESM2_{naming_d['experiment']}_{ens_name}_{exp.initial_condition_fake_year}.nc"
        if os.path.isfile(out_file_name) == False or overwrite:
            archive_fldr = f"{exp.dir_archive}/GKLT/initial_{exp.initial_conditions_name}/{case_identifier}"
            h_files = glob.glob(f"{archive_fldr}/{realm}/hist/*{h_identifier}*.nc")

            try:
                with xr.open_mfdataset(h_files, preprocess=preprocessor) as nc:
                    x = nc[variable]
                    initial_condition_year = x.time.dt.year.values[0]
                    x = x.assign_coords(time=xr.date_range(f"{exp.initial_condition_fake_year}-{exp.start_date_in_year}", periods=exp.n_days*exp.n_steps + 1)[1:])
                    if time_frequency == 'mon':
                        x = x.resample(time='ME').mean()
                    x = x.assign_coords(sim=case_identifier)
                    ds = xr.Dataset({variable_dict[variable]:x})
                    ds.attrs = nc.attrs
                    ds.attrs['initial_condition'] = initial_condition
                    ds.attrs['initial_condition_year'] = initial_condition_year
                    ds.attrs['compset'] = exp.compset
                    ds.attrs['readme'] = exp.git_repo
                    ds.attrs['preprocessing'] = preprocessing_attr
                    ds['time'].attrs['comment'] = f'This simulation represents the climate state of {exp.initial_conditions_name}. The year in the time axis can be ignored.'
                    os.makedirs(out_dir, exist_ok=True)
                    ds.to_netcdf(out_file_name)
            except:
                print('fail')

