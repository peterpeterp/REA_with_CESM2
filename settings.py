class configuration_parameters():
    def __init__(self):
        self.version=215
        self.res = 'f09_g17'
        self.mach = 'levante'
        self.compiler = 'intel'
        self.mpilib = 'openmpi'
        self.queue = 'default'
        self.walltime = '12:00:00'
        self.ntasks = 512
        self.ntasks_wav = 16

conf = configuration_parameters()

dir_scripts=f"/work/bb1152/u290372/cesm{conf.version}/cime/scripts"
dir_run=f"/scratch/u/u290372/cesm{conf.version}_output"
dir_repo=f"/home/u/u290372/projects/cesm215_peters_scripts/"
