#!/bin/python

import os,sys,subprocess,glob
import base64,hashlib

from settings import *

def run(command):
    command = command.replace('\n',' ')
    if args.verbose:
        print(f"\n---- subprocess call:\n{command}\n")
    return subprocess.run(command, shell=True, check=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--compset", type=str)

# this only controls where the output is stored
parser.add_argument("--dkrz_project_for_archive", type=str)

# full path where restart files are taken from
parser.add_argument("--parent_path", type=str)
# relative path where case will be stored
parser.add_argument("--case_path", type=str)
# relative path where a precompiled build can be found
parser.add_argument("--precompiled_path", type=str)

# ref case name is the case name under which the parent branch was run under
# if not specified, the last folder in the parent_path will be used
parser.add_argument("--ref_case_name", type=str)

parser.add_argument("--case_identifier", type=str)

parser.add_argument("--startdate", type=str)
parser.add_argument("--ndays", type=int)
parser.add_argument("--perturbation_type", type=str)
parser.add_argument("--perturbation_seed", type=int)
parser.add_argument("--perturbation_order_of_magnitude", type=float, default=-13)
parser.add_argument("--store_perturbation_file", action='store_true')

# directory where the user_nl files are found
parser.add_argument("--user_nl_file_directory", type=str)


parser.add_argument("--verbose", action='store_true')
parser.add_argument("--overwrite", action='store_true')
args = parser.parse_args()

# set archive dir
dir_archive=f"/work/{args.dkrz_project_for_archive}/u290372/cesm{conf.version}_archive"

assert args.perturbation_seed == 0 or args.perturbation_type is not None, "specify perturbation type if perturbation_seed != 0"

if args.case_identifier is None:
    args.case_identifier = str(args.perturbation_seed).zfill(3)
print(args.case_identifier)

#######################
# prepare directories #
#######################

if args.parent_path is not None and args.parent_path[-1] == '/':
    args.parent_path = args.parent_path[:-1]

if args.case_path is not None:
    # if this is given
    # example: first step of importance sampling -> new simulations can be stored in a subfolder for the experiment
    for directory in [dir_archive, dir_scripts, dir_run]:
        if os.path.isdir(f"{directory}/{args.case_path}") == False:
            os.system(f"mkdir -p {directory}/{args.case_path}")
else:
    # if not specified -> store the new run in the origin directory
    # all directories already exist
    args.case_path=f"{args.parent_path}/"

print(f"{args.parent_path}/rest/*/rpointer.atm")
if args.startdate is None:
    args.startdate = glob.glob(f"{args.parent_path}/rest/*/rpointer.atm")[0].split('/')[-2][:10]

############################
# Clean old case directory #
############################

if args.overwrite:
    if os.path.isdir(f"{dir_scripts}/{args.case_path}/{args.case_identifier}"):
        run(f"rm -rf {dir_scripts}/{args.case_path}/{args.case_identifier}")

##############
# Setup Case #
##############

os.chdir(f"{dir_scripts}/{args.case_path}")

command = f'''
{dir_scripts}/create_newcase --case {args.case_identifier}
--res {conf.res} --compset {args.compset} --mach {conf.mach} --compiler {conf.compiler}
--mpilib {conf.mpilib} --queue {conf.queue} --walltime {conf.walltime} --handle-preexisting-dirs u
--output-root {dir_run}/{args.case_path}
'''

run(command)

####################################
# Copy compiled bld from other run #
####################################

if args.precompiled_path is not None:
    print(f"copy bld/ and run/ from {dir_run}/{args.precompiled_path} to  {dir_run}/{args.case_path}/{args.case_identifier}")
    run(f"mkdir -p {dir_run}/{args.case_path}/{args.case_identifier}")
    run(f"rm -rf {dir_run}/{args.case_path}/{args.case_identifier}/bld")
    run(f"cp -al {dir_run}/{args.precompiled_path}/bld {dir_run}/{args.case_path}/{args.case_identifier}/")
    run(f"rsync -av --exclude '*.gz' --exclude '*.nc' {dir_run}/{args.precompiled_path}/run {dir_run}/{args.case_path}/{args.case_identifier}/")
    
##################
# configure case #
##################

os.chdir(args.case_identifier)

run(f"./xmlchange STOP_OPTION=ndays,STOP_N={args.ndays}")
run(f"./xmlchange RUNDIR={dir_run}/{args.case_path}/{args.case_identifier}/run")

run(f"./xmlchange NTASKS_CPL={conf.ntasks},NTASKS_ATM={conf.ntasks},NTASKS_LND={conf.ntasks},NTASKS_ICE={conf.ntasks},\
NTASKS_OCN={conf.ntasks},NTASKS_ROF={conf.ntasks},NTASKS_GLC={conf.ntasks}")

run(f"./xmlchange ROOTPE_CPL=0,ROOTPE_ATM=0,ROOTPE_OCN=0,ROOTPE_ICE=0,ROOTPE_LND=0,\
ROOTPE_WAV=0,ROOTPE_GLC=0,ROOTPE_ROF=0,ROOTPE_ESP=0")

run(f"./xmlchange NTASKS_WAV={conf.ntasks_wav}")
run(f"./xmlchange NTASKS_ESP=1")

run(f"./xmlchange RUN_TYPE=branch")

run(f"./xmlchange DOUT_S_ROOT={dir_archive}/{args.case_path}/{args.case_identifier}")

# this is the name used in scripts of the parent run
if args.ref_case_name is None:
    args.ref_case_name = args.parent_path.split('/')[-1]

run(f"./xmlchange RUN_REFCASE={args.ref_case_name}")
run(f"./xmlchange RUN_REFDATE={args.startdate}")
run(f"./xmlchange GET_REFCASE=FALSE")

######################################
# estimate time of job and set limit #
######################################
# has no effect apparanetly ?!?
# run(f"./xmlchange --subgroup case.run --id JOB_WALLCLOCK_TIME --val {args.JOB_WALLCLOCK_TIME}")



############################
# configure arhcive output #
############################

for component in ['cam', 'cice', 'cism', 'clm', 'cpl', 'mosart', 'pop', 'ww']:
    run(f'cp {args.user_nl_file_directory}/user_nl_{component} ./user_nl_{component}')

# # --- link postrun script ---
# # if [ $OUTPUT = default ]; then
#     # ./xmlchange POSTRUN_SCRIPT=/work/bb1152/u290372/cesm215_user_nl/default/postprocessing.sh
#     # ./xmlchange DATA_ASSIMILATION_SCRIPT=/work/bb1152/u290372/cesm215_user_nl/default/postprocessing.sh
# # fi

####################################
# save all command line arguments  #
# as well as the generating python #
# scripts in the case directory    #
####################################

for fl_name in [
    'branch_perturbed.py',
    'settings.py',
]:
    os.system(f"cp {dir_repo}/{fl_name} {dir_scripts}/{args.case_path}/{args.case_identifier}/{fl_name}")

with open(f"{dir_scripts}/{args.case_path}/{args.case_identifier}/command_line_arguments.txt", "w") as fl:
    for k,v in vars(args).items():
        fl.write(f"{k}: {v}\n")

##############
# setup case #
##############

run("./case.setup")

##########################################
# mv build to according folder structure #
##########################################

for fl in glob.glob(f"{args.parent_path}/rest/{args.startdate}-00000/*.r*.*"):
    fl = fl.split('/')[-1]
    run(f"ln -sf {args.parent_path}/rest/{args.startdate}-00000/{fl} {dir_run}/{args.case_path}/{args.case_identifier}/run/{fl}")

# # pointers should not be links
run(f"cp -va {args.parent_path}/rest/{args.startdate}-00000/rpointer.* {dir_run}/{args.case_path}/{args.case_identifier}/run/")

# # the cam.r file must not be a link as it will be modified!
run(f"rm {dir_run}/{args.case_path}/{args.case_identifier}/run/*cam.r.*")
run(f"cp -va {args.parent_path}/rest/{args.startdate}-00000/*cam.r.* {dir_run}/{args.case_path}/{args.case_identifier}/run/")

#############################################################
# manually modify the restart files with micro-pertubations #
#############################################################

print("*** Modify CAM restart file to mimic atmospheric perturbation")

os.chdir(f"{dir_run}/{args.case_path}/{args.case_identifier}/run")
cam_file=glob.glob("*cam.r.*")[0]

if args.perturbation_seed == 0:
    print("*********************")
    print("* NO Perturbation ! *")
    print("*********************")
else:
    #module purge
    run(f"module load cdo nco python3/2022.01-gcc-11.2.0")

    if args.perturbation_type == 'RH':
        '''
        Method used for boosting at ETH Zürich
        Standard value for args.perturbation_order_of_magnitude is -13
        Question: is the perturbation the same on all levels?
        '''

        print("Create random field with seed args.perturbation_seed")
        lon=int(subprocess.run(f"ncdump -h {cam_file} | grep  'lon = ' | tr -d 'lon = ' | tr -d ' ;'", shell=True, capture_output=True).stdout.strip())
        lat=int(subprocess.run(f"ncdump -h {cam_file} | grep  'lat = ' | tr -d 'lat = ' | tr -d ' ;'", shell=True, capture_output=True).stdout.strip())
        grid_file = "cam_grid.txt"
        with open(grid_file, 'w') as fl:
            fl.write("gridtype = lonlat\n")
            fl.write(f"xsize = {lon}\n")
            fl.write(f"ysize = {lat}\n")
            
        run(f"cdo -f nc random,{grid_file},{args.perturbation_seed} random.nc")
        run(f"cdo fldmean random.nc random_mean.nc")
        random_mean=float(subprocess.run(f"ncdump random_mean.nc | tail -2 | head -1 | tr -d ' ' | tr -d ';'", shell=True, capture_output=True).stdout.strip())
        run(f"cdo subc,{random_mean} random.nc random.tmp")
        run(f"mv random.tmp random.nc")
        run(f"rm random_mean.nc")

        print(f"Add random perturbation in the order of {args.perturbation_order_of_magnitude} to {cam_file}")
        run(f'cdo aexpr,"Q=Q*(1+10^({args.perturbation_order_of_magnitude})*random)" -merge {cam_file} random.nc {cam_file}.new')

        # remove var Q from original file
        run(f"ncks -x -v Q {cam_file} {cam_file}.woQ")
        run(f"mv {cam_file} {cam_file}.old")
        # extract var Q from new file
        run(f"ncks -v Q {cam_file}.new {cam_file}.Q")
        # merge both files
        run(f"ncks -A {cam_file}.woQ {cam_file}.Q")
        run(f"mv {cam_file}.Q {cam_file}")
        # clean up
        run(f"rm -f {cam_file}.woQ {cam_file}.old")

    if args.perturbation_type == 'PT_direct':
        '''
        reproducing Noyelle 
        Standard value for args.perturbation_order_of_magnitude is -4
        '''
        import numpy as np
        import xarray as xr

        # perturb PT by adding uniform random noise
        np.random.seed(args.perturbation_seed)
        nc = xr.open_dataset(f'{cam_file}')

        # create perturbation as amplitude*10**(perturbation_order_of_magnitude)*random
        nc['PT'] = nc['PT'] * (1 + np.random.uniform(-1,1,nc['PT'].shape) * 10**(args.perturbation_order_of_magnitude))
        nc.to_netcdf(f"{cam_file}.new")
        nc.close()

        # exchange and clean
        run(f"mv {cam_file} {cam_file}.old")
        run(f"mv {cam_file}.new {cam_file}")
        run(f"rm -f {cam_file}.old")


    if args.perturbation_type == 'PT_spectral':
        '''
        reproducing Ragone & Bouchet 2021
        Standard value for args.perturbation_order_of_magnitude is -4
        '''
        import numpy as np
        import xarray as xr

        # transform PT to spectral coordinates
        run(f"cdo gp2sp -remaplaf,n96 -selname,PT {cam_file} PT_spectral.nc")

        # perturb PT by adding uniform random noise
        np.random.seed(args.perturbation_seed)
        nc = xr.open_dataset('PT_spectral.nc')

        # introduce perturbation as amplitude*10**(perturbation_order_of_magnitude)*random
        # generate random numbers to perturb amplitudes
        random_real = np.random.uniform(-1,1,nc['PT'].shape[1])
        random_img = np.random.uniform(-1,1,nc['PT'].shape[1])

        # these random numbers are going to be used at all levels
        for lev in nc.lev.values:
            nc['PT'].loc[lev,:,0] *=  random_real * 10**(args.perturbation_order_of_magnitude)
            nc['PT'].loc[lev,:,1] *=  random_img * 10**(args.perturbation_order_of_magnitude)

        # do not perturb first mode
        nc['PT'][:,0,:] = 0

        nc.to_netcdf('perturbation_spectral.nc')
        nc.close()

        # retransform to original grid
        run(f"cdo remaplaf,{cam_file} -sp2gp -chname,PT,PT_perturbation perturbation_spectral.nc perturbation.nc")

        # add perturbation
        run(f'cdo aexpr,"PT=PT+PT_perturbation" -merge {cam_file} perturbation.nc {cam_file}.new')

        # remove var PT from original file
        run(f"ncks -x -v PT {cam_file} {cam_file}.woPT")
        run(f"mv {cam_file} {cam_file}.old")
        # extract var PT from new file
        run(f"ncks -v PT {cam_file}.new {cam_file}.PT")
        # merge both files
        run(f"ncks -A {cam_file}.woPT {cam_file}.PT")
        run(f"mv {cam_file}.PT {cam_file}")
        # clean up
        run(f"rm -f {cam_file}.woPT {cam_file}.old")

if args.store_perturbation_file:
    # keep the cam_file file for reference
    logs=f"{dir_archive}/{args.case_path}/{args.case_identifier}/logs"
    run(f"mkdir -p {logs}")
    run(f"cp -av {cam_file} {logs}/")

##############
# build case #
##############

os.chdir(f"{dir_scripts}/{args.case_path}/{args.case_identifier}")


if args.precompiled_path is not None:
    build_xml = open(f"{dir_scripts}/{args.precompiled_path}/env_build.xml", "r").read()
    l = []
    for line in build_xml.split('\n'):
        if '<entry id="CIME_OUTPUT_ROOT" value="' in line:
            l.append(f'    <entry id="CIME_OUTPUT_ROOT" value="{dir_run}/{args.case_path}/">')
        else:
            l.append(line)
    with open("./env_build.xml", "w") as fl:
        fl.write('\n'.join(l))
    # run(f"cp {dir_scripts}/{args.precompiled_path}/env_build.xml ./env_build.xml")    
else:
    run("./case.build")    

###############
# submit case #
###############

run("./case.submit")


with open(f"generating_command.log", 'w') as fl:
    fl.write(f"command line arguments:\n")
    for k,v in args.__dict__.items():
        fl.write(f"{k.ljust(30)}{v}\n")

    fl.write(f"\n")
    fl.write(f"global settings arguments:\n")
    for k,v in conf.__dict__.items():
        fl.write(f"{k.ljust(30)}{v}\n")


'''
for pert in {1..10}; do python /home/u/u290372/projects/cesm215_peters_scripts/branch_perturbed.py  
--compset BSSP370cmip6 --ndays 15 --perturbation_seed $(($pert+11)) --perturbation_order_of_magnitude -$pert --perturbation_type PT_direct  
--parent_path SH_enso_stratVort/b.e215.BSSP370cmip6.f09_g17.0400.fE.2020.ens000.branch/2022-01-01_to_2022-09-30/000/000/002/000 
--precompiled SH_enso_stratVort/b.e215.BSSP370cmip6.f09_g17.0400.fE.2020.ens000.branch/2022-01-01_to_2022-09-30/000; done;
'''