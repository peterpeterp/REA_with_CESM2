import os,sys,subprocess,glob,importlib
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd

from main_launcher import launch_handler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--dry_run", action='store_true')
    parser.add_argument("--step", type=int)
    command_line_arguments = parser.parse_args()

    # load configuration settings
    experiment_configuration = importlib.import_module(f"experiments.{command_line_arguments.experiment}")
    
    # finalize experiment configuration
    from experiments.experiment import experiment
    exp = experiment(experiment_configuration.config)

    launcher = launch_handler(
        exp, 
        verbose=command_line_arguments.verbose, 
        dry_run=command_line_arguments.dry_run, 
        relaunch_cases_which_are_unclear=False, 
        relaunch_after_completion=False
        )

    launcher.clean_run_directories_of_step_X(command_line_arguments.step)
