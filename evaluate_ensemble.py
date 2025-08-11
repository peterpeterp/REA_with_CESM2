import os,sys,subprocess,glob,cftime,importlib,pickle,itertools
from datetime import datetime
import xarray as xr
import numpy as np
sys.path.append('../')

from ensembles.ensemble_GKLT import ensemble_GKLT,get_weight_for_selection

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--project_path", type=str)
parser.add_argument("--experiment_identifiers", nargs='+', default=[f"c{i}" for i in range(1,6)] + [f"p{i}" for i in range(1,6)] + ['c1_initial', 'p1_initial'])
parser.add_argument("--variable_name_in_filename", type=str)
parser.add_argument("--variable_name_in_file", type=str)
parser.add_argument("--overwrite", action='store_true')
command_line_arguments = parser.parse_args()

sys.path.append(command_line_arguments.project_path)
from experiment_configuration.experiment import experiment

for experiment_identifier in command_line_arguments.experiment_identifiers:
    print(experiment_identifier)
    exp = experiment(importlib.import_module(f"experiment_configuration.{experiment_identifier}").config)

    exp_new_name = ''.join(exp.experiment_identifier.split('_')[0][1:])

    naming_d = {
        "project": 'REA_output',
        "product": exp.product_name,
        "institute": 'NCAR',
        "model": 'CESM2',
        "experiment" : f"{exp.initial_conditions_name}-x{exp_new_name}",
        "realm": "meta",
    }
    out_dir = '/'.join([exp.dir_work] + [v for k,v in naming_d.items()])
    os.makedirs(out_dir, exist_ok=True)

    obs = xr.open_mfdataset(f"{exp.dir_work}/REA_output/{exp.product_name}/NCAR/CESM2/{exp.initial_conditions_name}-x{exp_new_name}/meta/obs/*/*", concat_dim='sim', combine='nested')['obs'].load()

    ens = ensemble_GKLT(exp)
    ens.evaluate_weights_and_probabilities(obs)
    print(ens._prob)
    print(ens._weight_from_algo)
    xr.Dataset({'probability':ens._prob}).to_netcdf(f"{out_dir}/probability_season_{naming_d['experiment']}.nc")
    xr.Dataset({'weight':ens._weight_from_algo}).to_netcdf(f"{out_dir}/weight_season_{naming_d['experiment']}.nc")