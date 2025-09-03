import os,sys,subprocess,glob,cftime,importlib,pickle,itertools
from datetime import datetime,timedelta
import xarray as xr
import numpy as np
sys.path.append('../')

from ensembles.ensemble_GKLT import ensemble_GKLT


realm_dict = {
    'atm' : 'atmos',
    'lnd' : 'land',
}

variable_dict = {
    'obs' : 'obs',
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
    'LHFLX' : 'hfls',
    'SHFLX' : 'hfss',
}


def open_rea(exp, sim_name, realm, h_identifier, variable, preprocessor, end_step=None):
    l = []
    for step,sim_identifier_in_step in enumerate(sim_name.split('.')[1:]):
        h_files = glob.glob(f"{exp.dir_archive_post}/step{step}/{exp.experiment_identifier}_{sim_identifier_in_step}/{realm}/hist/*{h_identifier}*.nc")
        if len(h_files) == 1:
            with xr.open_mfdataset(h_files[0]) as nc:
                if preprocessor is not None:
                    nc = preprocessor(nc, variable)
                l.append(nc[variable])
    if len(l) == end_step:
        x = xr.concat(l, dim='time')
        
    return x, nc.attrs

def open_rea_legacy(exp, sim_name, realm, h_identifier, variable, preprocessor, end_step=None):
    todos = [['/'.join(sim_name.split('/')[:step])] for step in range(1,end_step+1)]
    l = []
    for step in range(1,end_step+1):
        _sim_name_of_step_ = '/'.join(sim_name.split('/')[:step])
        h_files = glob.glob(f"{exp.dir_archive_post}/{_sim_name_of_step_}/{realm}/hist/*{h_identifier}*.nc")
        if len(h_files) == 1:
            with xr.open_mfdataset(h_files[0]) as nc:
                if preprocessor is not None:
                    nc = preprocessor(nc, variable)
                l.append(nc[variable])

    if len(l) == end_step:
        x = xr.merge(l)[variable].sortby('time')

    return x, nc.attrs

def open_initial(exp, sim_name, realm, h_identifier, variable, preprocessor):
    archive_fldr = f"{exp.dir_archive}/GKLT/initial_{exp.initial_conditions_name}_{exp.start_date_in_year}/{sim_name}"
    h_files = glob.glob(f"{archive_fldr}/{realm}/hist/*{h_identifier}*.nc")

    with xr.open_mfdataset(h_files) as nc:
        if preprocessor is not None:
            nc = preprocessor(nc, variable)
        return nc[variable], nc.attrs

def open_initial_before(exp, sim_name, realm, h_identifier, variable, preprocessor):
    initial_archive = [s for s in exp.initial_conditions if s.split('/')[-1][:4] == sim_name.split('_')[-1] and s.split('.fE.')[0].split('.')[-1] == sim_name.split('_')[0] ][0]
    if exp.initial_conditions_name == 'piControl':
        initial_before_archive = initial_archive.split('/branch/')[0]
        initial_year = initial_archive.split('/')[-1][:4]
    else:
        initial_before_archive = '/'.join(initial_archive.split('/')[:-1])
        initial_year = int(initial_archive.split('/')[-1][:4])

    h_files = glob.glob(f"{initial_before_archive}/atm/hist/*{h_identifier}.{initial_year}*")
    with xr.open_mfdataset(h_files) as nc:
        if preprocessor is not None:
            nc = preprocessor(nc, variable)
        i_first_day = nc.time.loc[:f"{str(nc.time.dt.year.values[0]).zfill(4)}-{exp.start_date_in_year}"].shape[0]+1
        return nc[variable][:i_first_day], nc.attrs



def extract(
    experiment_identifier,
    exp,
    variable = 'Z500',
    realm = 'atm',
    h_identifier = 'h1',
    time_frequency = 'day',
    preprocessing_module = None,
    end_step = None,
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

    if variable == 'obs':
        naming_d['time_frequency'] = ''
        naming_d['realm'] = 'meta'

    if end_step is None:
        end_step = exp.n_steps
    else:
        naming_d['experiment'] += f"-step{end_step}"

    if exp.ensemble_type == 'initial' or exp.ensemble_type == 'before':
        naming_d['experiment'] = f"{exp.initial_conditions_name}-initial"
        trajectory_names = [ini.split('.')[-4] + '_' + ini.split('/')[-1].split('-')[0] for ini in exp.initial_conditions]
    elif exp.ensemble_type == 'rea_legacy':
        exp_new_name = ''.join(exp.experiment_identifier.split('_')[0][1:])
        naming_d['experiment'] = f"{exp.initial_conditions_name}-x{exp_new_name}"
        ens = ensemble_GKLT(exp)
        trajectory_names = sorted([s for s in ens._sim_names if len(s.split('/')) == end_step])
    elif exp.ensemble_type == 'rea':
        exp_new_name = ''.join(exp.experiment_identifier.split('_')[0][1:])
        naming_d['experiment'] = f"{exp.initial_conditions_name}-x{exp_new_name}"
        ens = ensemble_GKLT(exp)
        trajectory_names = ens._sim_names
    else:
        assert False, 'need ensemble_type'


    if 'before' in experiment_identifier:
        naming_d['variable'] += '-before'

    print(naming_d)


    for i,sim_name in enumerate(trajectory_names):
        ens_name = f"ens{str(i+1).zfill(3)}"
        out_dir = '/'.join([exp.dir_work] + [v for k,v in naming_d.items()] + [ens_name])
        #first_date = f"{exp.initial_condition_fake_year}-{exp.start_date_in_year}"
        #last_date = str(datetime.strptime(first_date, "%Y-%m-%d") + timedelta(days=exp.n_days * end_step))[:10]
        out_file_name = f"{out_dir}/{naming_d['variable']}_{naming_d['time_frequency']}_CESM2_{naming_d['experiment']}_{ens_name}_{exp.initial_condition_fake_year}.nc"
        print(sim_name, out_file_name)
        if os.path.isfile(out_file_name) == False or overwrite:
            if exp.ensemble_type == 'before':
                x, attrs = open_initial_before(exp, sim_name, realm, h_identifier, variable, preprocessor)
                x = x.assign_coords(sim=sim_name)
                x = x.assign_coords(time=xr.date_range(f"{exp.initial_condition_fake_year}-01-01", periods=len(x.time.values)))
            elif exp.ensemble_type == 'initial':
                x, attrs = open_initial(exp, sim_name, realm, h_identifier, variable, preprocessor)
                x = x.assign_coords(sim=sim_name)
                x = x.assign_coords(time=xr.date_range(f"{exp.initial_condition_fake_year}-{exp.start_date_in_year}", periods=exp.n_days*end_step + 1)[1:])
            elif exp.ensemble_type == 'rea_legacy':
                x, attrs = open_rea_legacy(exp, sim_name, realm, h_identifier, variable, preprocessor, end_step)
                x = x.assign_coords(sim=sim_name)
                x = x.assign_coords(time=xr.date_range(f"{exp.initial_condition_fake_year}-{exp.start_date_in_year}", periods=exp.n_days*end_step + 1)[1:])
            elif exp.ensemble_type == 'rea':
                x, attrs = open_rea(exp, sim_name, realm, h_identifier, variable, preprocessor, end_step)
                x = x.assign_coords(sim=sim_name)
                x = x.assign_coords(time=xr.date_range(f"{exp.initial_condition_fake_year}-{exp.start_date_in_year}", periods=exp.n_days*end_step + 1)[1:])
            else:
                assert False, 'need ensemble_type'

            # monthly average
            if time_frequency == 'mon':
                x = x.resample(time='ME').mean()
            
            ds = xr.Dataset({variable_dict[variable]:x})
            ds.attrs = attrs
            ds.attrs['simulation_name'] = sim_name
            '''
            I think the initial condition in the atributes is wrong
            need to check
            '''

            ds.attrs['initial_condition'] = exp.initial_conditions[i]
            ds.attrs['initial_condition_year'] = exp.initial_conditions[i].split('/')[-1].split('-')[0]
            ds.attrs['compset'] = exp.compset
            ds.attrs['readme'] = exp.git_repo
            ds.attrs['preprocessing'] = preprocessing_attr
            ds['time'].attrs['comment'] = f'This simulation represents the climate state of {exp.initial_conditions_name}. The year in the time axis can be ignored.'

            os.makedirs(out_dir, exist_ok=True)
            ds.to_netcdf(out_file_name)

