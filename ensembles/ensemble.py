import os,sys,glob,cftime
import xarray as xr
import numpy as np


class ensemble():
    def __init__(self, exp):
        self._exp = exp
        self._dir = f"{exp.dir_work}/GKLT/{self._name}"
        if os.path.isdir(self._dir) == False:
            os.system(f"mkdir {self._dir}")
        self._trajectories = {}
    
        d = f"{self._dir}/data"
        if os.path.isdir(d) == False:
            os.system(f'mkdir -p {d}')

        d = f"{self._dir}/labels"
        if os.path.isdir(d) == False:
            os.system(f'mkdir -p {d}')