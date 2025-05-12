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
    'RHREFHT' : 'rhs',
    'U200' : 'ua200',
    'U500' : 'ua500',
    'V500' : 'va500',
    'Z500' : 'zg500',
    'PSL' : 'psl',
    'SST' : 'tos',
    'ICEFRAC' : 'sic',
    'SOILWATER_10CM' : 'mrsos',
    'NEP' : 'nep',
    'pr' : 'pr',
}

def open_rea(exp, sim_name, realm, h_identifier, variable, preprocessor):
    todos = [['/'.join(sim_name.split('/')[:step])] for step in range(1,exp.n_steps+1)]
    l = []
    for step in range(1,exp.n_steps+1):
        _sim_name_of_step_ = '/'.join(sim_name.split('/')[:step])
        h_files = glob.glob(f"{exp.dir_archive_post}/{_sim_name_of_step_}/{realm}/hist/*{h_identifier}*.nc")
        if len(h_files) == 1:
            with xr.open_mfdataset(h_files[0], preprocess=preprocessor) as nc:
                l.append(nc[variable])

    if len(l) == exp.n_steps:
        x = xr.merge(l)[variable].sortby('time')

    return x, nc.attrs

def open_initial(exp, sim_name, realm, h_identifier, variable, preprocessor):
    archive_fldr = f"{exp.dir_archive}/GKLT/initial_{exp.initial_conditions_name}_{exp.start_date_in_year}/{sim_name}"
    h_files = glob.glob(f"{archive_fldr}/{realm}/hist/*{h_identifier}*.nc")
    with xr.open_mfdataset(h_files, preprocess=preprocessor) as nc:
        return nc[variable], nc.attrs


def extract(
    experiment_identifier,
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
        "experiment" : 'missing',
        "time_frequency": time_frequency,
        "realm": realm_dict[realm],
        "variable": variable_dict[variable]+name_addition,
    }

    if variable in ['SST']:
        naming_d['realm'] = 'ocean'

    if variable in ['ICEFRAC']:
        naming_d['realm'] = 'seaIce'

    if 'initial' in experiment_identifier:
        naming_d['experiment'] = f"{exp.initial_conditions_name}-initial"
        trajectory_names = [ini.split('.')[-4] + '_' + ini.split('/')[-1].split('-')[0] for ini in exp.initial_conditions]
    else:
        naming_d['experiment'] = f"{exp.initial_conditions_name}-x{exp.experiment_identifier[1]}"
        ens = ensemble_GKLT(exp)
        trajectory_names = sorted([s for s in ens._sim_names if len(s.split('/')) == ens._exp.n_steps])



    for i,sim_name in enumerate(trajectory_names):
        ens_name = f"ens{str(i+1).zfill(3)}"
        out_dir = '/'.join([exp.dir_work] + [v for k,v in naming_d.items()] + [ens_name])
        out_file_name = f"{out_dir}/{variable_dict[variable]+name_addition}_{time_frequency}_CESM2_{naming_d['experiment']}_{ens_name}_{exp.initial_condition_fake_year}.nc"
        print(out_file_name)
        if os.path.isfile(out_file_name) == False or overwrite:
            if 'initial' in experiment_identifier:
                x, attrs = open_initial(exp, sim_name, realm, h_identifier, variable, preprocessor)
                x = x.assign_coords(sim=sim_name)
            else:
                x, attrs = open_rea(exp, sim_name, realm, h_identifier, variable, preprocessor)
                x = x.assign_coords(sim=sim_name)

            # time axis with same year
            x = x.assign_coords(time=xr.date_range(f"{exp.initial_condition_fake_year}-{exp.start_date_in_year}", periods=exp.n_days*exp.n_steps + 1)[1:])

            # monthly average
            if time_frequency == 'mon':
                x = x.resample(time='ME').mean()
            
            ds = xr.Dataset({variable_dict[variable]:x})
            ds.attrs = attrs
            ds.attrs['simulation_name'] = sim_name
            ds.attrs['initial_condition'] = exp.initial_conditions[i]
            ds.attrs['initial_condition_year'] = exp.initial_conditions[i].split('/')[-1].split('-')[0]
            ds.attrs['compset'] = exp.compset
            ds.attrs['readme'] = exp.git_repo
            ds.attrs['preprocessing'] = preprocessing_attr
            ds['time'].attrs['comment'] = f'This simulation represents the climate state of {exp.initial_conditions_name}. The year in the time axis can be ignored.'

            os.makedirs(out_dir, exist_ok=True)
            ds.to_netcdf(out_file_name)

