import os,glob
import xarray as xr


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--search_path", type=str)
parser.add_argument("--overwrite", action='store_true')
command_line_arguments = parser.parse_args()

 # '/work/bb1152/u290372/REA_output/heat_wEU_JJA/NCAR/CESM2/*/day/atmos/stream500/*/*'

for fl in sorted(glob.glob(command_line_arguments.search_path)):
    print(fl)
    out_fl = fl.replace('day','mon')
    if os.path.isfile(out_fl) == False or command_line_arguments.overwrite:
        os.makedirs('/'.join(out_fl.split('/')[:-1]), exist_ok=True)
        with xr.open_dataset(fl) as nc_in:
            nc_in.resample(time='ME').mean().to_netcdf(out_fl)


'''
/work/bb1152/u290372/REA_output/heat_wEU_JJA/NCAR/CESM2/*/day/atmos/ua500/*/*

'''