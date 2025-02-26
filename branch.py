#!/bin/python

import os,sys,subprocess,glob
from datetime import datetime

from settings import *

def run(command):
    command = command.replace('\n',' ')
    if args.verbose:
        print(f"\n---- subprocess call:\n{command}\n")
    return subprocess.run(command, shell=True, check=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--compset", type=str)
parser.add_argument("--project", type=str, default='bb1445')
parser.add_argument("--parent", type=str)
parser.add_argument("--case_name_addon", type=str)
parser.add_argument("--startdate", type=str)
parser.add_argument("--enddate", type=str)
parser.add_argument("--precompiled", type=str)
parser.add_argument("--output", type=str, default="default")
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--overwrite", action='store_true')
args = parser.parse_args()

# set archive dir
dir_archive=f"/work/{args.project}/u290372/cesm{conf.version}_archive"

# identify parent
assert (args.parent is not None), "give parent"

if args.parent is not None:
    dir_case=f"{args.parent}/branch/"

for directory in [dir_archive, dir_scripts, dir_run]:
    os.system(f"mkdir -p {directory}/{dir_case}")

# get start date and end date
assert (args.enddate is not None), "give enddate"

if args.startdate is None:
    args.startdate = glob.glob(f"{dir_archive}/{args.parent}/rest/*/rpointer.atm")[0].split('/')[-2][:10]

if args.enddate is not None:
    stop_condition = 'ndays'
    # calculate ndays
    startdate = datetime.strptime(args.startdate, '%Y-%m-%d')
    enddate = datetime.strptime(args.enddate, '%Y-%m-%d')
    stop_N = (enddate - startdate).total_seconds() / (24*60*60)
    stop_N = int(stop_N)
    # CESM2 has no leap years
    leap_year = (datetime(startdate.year, 3, 1) - datetime(startdate.year, 2, 1)).total_seconds() / (24*60*60) == 29
    if enddate.month > 2 and leap_year:
        stop_N -= 1

print(args)

# case name
case_name=f"{args.startdate}_to_{args.enddate}"
if args.case_name_addon is not None:
    case_name = f"{args.case_name_addon}.{case_name}"
print(case_name)

############################
# Clean old case directory #
############################

if args.overwrite:
    if os.path.isdir(f"{dir_scripts}/{dir_case}/{case_name}"):
        run(f"rm -rf {dir_scripts}/{dir_case}/{case_name}")

##############
# Setup Case #
##############

os.chdir(f"{dir_scripts}/{dir_case}")

command = f'''
{dir_scripts}/create_newcase --case {case_name}
--res {conf.res} --compset {args.compset} --mach {conf.mach} --compiler {conf.compiler}
--mpilib {conf.mpilib} --queue {conf.queue} --walltime {conf.walltime} --handle-preexisting-dirs u
--output-root {dir_run}/{dir_case}
'''

run(command)

####################################
# save all command line arguments  #
# as well as the generating python #
# scripts in the case directory    #
####################################

for fl_name in [
    'branch.py',
    'settings.py',
]:
    os.system(f"cp {dir_repo}/{fl_name} {dir_scripts}/{dir_case}/{case_name}/{fl_name}")

with open(f"{dir_scripts}/{dir_case}/{case_name}/command_line_arguments.txt", "w") as fl:
    for k,v in vars(args).items():
        fl.write(f"{k}: {v}\n")

####################################
# Copy compiled bld from other run #
####################################

if args.precompiled is not None:
    print(f"copy bld/ and run/ from {args.precompiled} to  {dir_run}/{dir_case}/{case_name}")
    run(f"mkdir -p {dir_run}/{dir_case}/{case_name}")
    run(f"rm -rf {dir_run}/{dir_case}/{case_name}/bld")
    run(f"cp -al {dir_run}/{args.precompiled}/bld {dir_run}/{dir_case}/{case_name}/")
    run(f"rsync -av --exclude '*.gz' --exclude '*.nc' {dir_run}/{args.precompiled}/run {dir_run}/{dir_case}/{case_name}/")
    

##################
# configure case #
##################

os.chdir(case_name)

run(f"./xmlchange STOP_OPTION={stop_condition},STOP_N={stop_N}")
run(f"./xmlchange DOUT_S_ROOT={dir_archive}/{dir_case}/{case_name}")
run(f"./xmlchange RUNDIR={dir_run}/{dir_case}/{case_name}/run")

run(f"./xmlchange NTASKS_CPL={conf.ntasks},NTASKS_ATM={conf.ntasks},NTASKS_LND={conf.ntasks},NTASKS_ICE={conf.ntasks},\
NTASKS_OCN={conf.ntasks},NTASKS_ROF={conf.ntasks},NTASKS_GLC={conf.ntasks}")

run(f"./xmlchange ROOTPE_CPL=0,ROOTPE_ATM=0,ROOTPE_OCN=0,ROOTPE_ICE=0,ROOTPE_LND=0,\
ROOTPE_WAV=0,ROOTPE_GLC=0,ROOTPE_ROF=0,ROOTPE_ESP=0")

run(f"./xmlchange NTASKS_WAV={conf.ntasks_wav}")
run(f"./xmlchange NTASKS_ESP=1")

run(f"./xmlchange RUN_TYPE=branch")
run(f"./xmlchange RUN_REFCASE={args.parent.split('/')[-1]}")
run(f"./xmlchange RUN_REFDATE={args.startdate}")
run(f"./xmlchange GET_REFCASE=FALSE")

############################
# configure arhcive output #
############################

for component in ['cam', 'cice', 'cism', 'clm', 'cpl', 'mosart', 'pop', 'ww']:
    run(f'cp ~/projects/cesm215_peters_scripts/cesm215_user_nl/{args.output}/user_nl_{component} ./user_nl_{component}')

##############
# setup case #
##############

run("./case.setup")

##########################################
# mv build to according folder structure #
##########################################

for fl in glob.glob(f"{dir_archive}/{args.parent}/rest/{args.startdate}-00000/*.r*.*"):
    fl = fl.split('/')[-1]
    run(f"ln -sf {dir_archive}/{args.parent}/rest/{args.startdate}-00000/{fl} {dir_run}/{dir_case}/{case_name}/run/{fl}")

# # pointers should not be links
run(f"cp -va {dir_archive}/{args.parent}/rest/{args.startdate}-00000/rpointer.* {dir_run}/{dir_case}/{case_name}/run/")


##############
# build case #
##############

os.chdir(f"{dir_scripts}/{dir_case}/{case_name}")


if args.precompiled is not None:
    build_xml = open(f"{dir_scripts}/{args.precompiled}/env_build.xml", "r").read()
    l = []
    for line in build_xml.split('\n'):
        if '<entry id="CIME_OUTPUT_ROOT" value="' in line:
            l.append(f'    <entry id="CIME_OUTPUT_ROOT" value="{dir_run}/{dir_case}/">')
        else:
            l.append(line)
    with open("./env_build.xml", "w") as fl:
        fl.write('\n'.join(l))
    # run(f"cp {dir_scripts}/{args.experiment}/{args.origin}/{args.precompiled}/env_build.xml ./env_build.xml")    
else:
    run("./case.build")   
  

###############
# submit case #
###############

run("./case.submit")


