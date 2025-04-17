import os,sys,subprocess,glob,cftime,importlib,pickle,itertools
from datetime import datetime
import xarray as xr
import numpy as np
sys.path.append('../')

from ensembles.ensemble_GKLT import ensemble_GKLT
from data_extractor import extract

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--project_path", type=str)
parser.add_argument("--variable", type=str)
parser.add_argument("--realm", type=str)
parser.add_argument("--h_identifier", type=str)
parser.add_argument("--time_frequency", type=str, default='day')
parser.add_argument("--preprocessing", type=str, default='')
parser.add_argument("--experiment_identifiers", nargs='+', default=[f"c{i}" for i in range(1,6)] + [f"p{i}" for i in range(1,6)] + ['c1_initial', 'p1_initial'])
parser.add_argument("--overwrite", action='store_true')
command_line_arguments = parser.parse_args()

sys.path.append(command_line_arguments.project_path)
from experiment_configuration.experiment import experiment

for k,v in vars(parser.parse_args()).items():
    globals()[k] = v

if preprocessing == '':
    preprocessing_module = None
else:
    preprocessing_module = importlib.import_module(f"data_extraction.preprocessing.{preprocessing}")

for experiment_identifier in experiment_identifiers:
    print(experiment_identifier)
    # load experiment configuration settings
    if 'initial' in experiment_identifier:
        exp = experiment(importlib.import_module(f"experiment_configuration.{experiment_identifier.replace('_initial','')}").config)
    else:
        exp = experiment(importlib.import_module(f"experiment_configuration.{experiment_identifier}").config)

    extract(
        experiment_identifier, 
        exp,
        variable = variable,
        realm = realm,
        h_identifier = h_identifier,
        time_frequency = time_frequency,
        preprocessing_module = preprocessing_module,
        overwrite = overwrite,
    )