#!/bin/bash

NAME=""

# Loop through all the arguments passed to the script
prev_param=""
for variable in "$@"; do
   if [[ "$prev_param" == "--variable" ]]; then
      NAME=$variable
      break
   fi

   # Save current param as previous for the next iteration
   prev_param="$variable"
done

# Loop through all the arguments passed to the script
prev_param=""
for preprocessing in "$@"; do
   if [[ "$prev_param" == "--preprocessing" ]]; then
      NAME=$variable$preprocessing
      break
   fi

   # Save current param as previous for the next iteration
   prev_param="$preprocessing"
done


sbatch -J extract_$NAME slurm_job.sh ~/.conda/envs/py_imps/bin/python3 extract.py "$@"