from ensembles.ensemble import * 

import multiprocessing
import pandas as pd

import matplotlib.pyplot as plt

from anytree import NodeMixin, Node, RenderTree, AsciiStyle, findall

class simulationBase(object):
    dummy = None

class simulation_tree(simulationBase, NodeMixin):  # Add Node feature
    def __init__(self, full_name, forest):
        self.full_name = full_name
        self.name = full_name.split('/')[-1]
        self._data = {}
        if len(full_name.split('/')) > 1:
            self.parent = forest['/'.join(full_name.split('/')[:-1])]
        else:
            self.parent = None

class ensemble_GKLT(ensemble):
    def __init__(self, exp):
        self._name = exp.experiment_name
        super().__init__(exp)
        self.get_sim_names()
        self.build_forest()

        # Ris
        self._mean_scores = np.array([
            pd.read_table(f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step}_evaluation.csv", sep=',')['score'].values.mean()
            for step in range(self._exp.n_steps)
        ])

    ###################################
    # Get data structure of ensemble  #
    # build a forest                  #
    ###################################

    def get_sim_names(self):
        file_name = f"{self._exp.dir_out}/sim_names.txt"
        if os.path.isfile(file_name) == False:
            self._sim_names = []
            for step in range(self._exp.n_steps):
                sim_paths = sorted([p for p in glob.glob(f"{self._exp.dir_archive_post}/*/{'/*'*step}/atm")])
                self._sim_names += [p.replace(f"{self._exp.dir_archive_post}/","").replace('/atm','') for p in sim_paths]

            self._sim_names = np.array(self._sim_names)
            with open(file_name, 'w') as fl:
                fl.write(';'.join(self._sim_names))
        else:
            self._sim_names = open(file_name, 'r').read().split(';')

    def build_forest(self):
        self._forest = {}
        for sim_name in self._sim_names:
            self._forest[sim_name] = simulation_tree(sim_name, self._forest)

    def fill_forest_with_pieces(self, var_name):
        for sim_name in self._forest.keys():
            self._forest[sim_name]._data[var_name] = xr.open_dataset(f"{self._exp.dir_archive_post}/{sim_name}/post/{var_name}.nc")[var_name]         

    def print_tree(self, initial_name, show_uncomplete=False):
        class bcolors:
            HEADER = '\033[95m'
            OKBLUE = '\033[94m'
            OKCYAN = '\033[96m'
            OKGREEN = '\033[92m'
            WARNING = '\033[93m'
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'

        sims = sorted([s for s in self._sim_names if len(s.split('/')) == self._exp.n_steps])
        sims = np.array([s.split('/') for s in sims])

        for pre, fill, node in RenderTree(self._forest[initial_name]):
            p = node
            l = [p.name]
            step = 1
            while p.parent is not None:
                p = p.parent
                l += [f'{p.name}']
                step += 1
            s = '/'.join(l[::-1])
            if step == 18:
                w = f"  -> {float(self._weight.loc[s]):.2f}"
            else:
                w = ''
            if sum([s in '/'.join(sims[i,:step]) for i in range(sims.shape[0])]) > 0:
                print(bcolors.WARNING + f"{pre}{node.name}{w}" + bcolors.ENDC)
            else:
                if show_uncomplete:
                    print(bcolors.OKCYAN + "%s%s" % (pre, node.name) + bcolors.ENDC)

    def get_weights_uniqueness(self):
        '''
        weights based on uniqueness
        '''
        sims = sorted([s for s in self._sim_names if len(s.split('/')) == self._exp.n_steps])
        weight = xr.DataArray(dims=['sim','step'], coords=dict(sim=sims,step=np.arange(0,self._exp.n_steps,1,'int')))
        weight_daily = xr.DataArray(
            dims=['sim','step','time'], 
            coords=dict(sim=sims,step=np.arange(0,self._exp.n_steps,1,'int'),time=np.arange(0,self._exp.n_days,1,'int'))
        )
        sims = np.array([s.split('/') for s in sims])

        for step in weight.step.values:
            for v in np.unique(sims[:,step]):
                same = (sims[:,step] == v)
                weight[same, step] = 1 / same.sum()
                weight_daily[same, step] = 1 / same.sum()

        self._weight = weight.mean('step')
        self._weight_stepwise = weight
        self._weight_daily = xr.DataArray(
            weight_daily.values.reshape((self._exp.n_members, self._exp.n_steps * self._exp.n_days)),
            dims = ['sim','time'],
            coords = dict(sim=weight.sim, time=np.arange(0,self._exp.n_days * self._exp.n_steps,1,'int'))
        )
        return weight

    ##################
    # First analysis #
    ##################

    def calc_scores_Ris_etc(self):
        self._abs = self._trajectories['obs'].mean('time')
        self.calculate_time_sum_over_each_step()
        self.calculate_scores()
        self.get_probabilities()

    def calculate_time_sum_over_each_step(self):
        x = xr.DataArray(
            self._trajectories['obs'].values.reshape((-1, self._exp.n_steps, self._exp.n_days)),
            dims=['sim','step','time'],
            coords=dict(sim=self._trajectories['obs'].sim, step=np.arange(0,self._exp.n_steps,1,'int'), time=np.arange(0,self._exp.n_days,1,'int'))
        )
        self._time_sum_over_each_step = x.sum('time')

    def calculate_scores(self):
        self._scores = np.exp(self._exp.k * self._time_sum_over_each_step)

    def get_probabilities(self):
        # what about last step??
        self._p = self._trajectories['obs'].mean('time').copy() * np.nan
        self._p[:] = np.array(
            [
                np.product(self._mean_scores[:-1]) / (np.product(self._scores[i,:-1]) * self._exp.n_members)
                for i in range(self._exp.n_members)
            ]
        )
        self._prob = self._trajectories['obs'].mean('time').copy() * np.nan
        self._prob[:] = np.array([np.sum((self._abs >= a).astype(float) * self._p) for a in self._abs]) 

    def ra(self, x, thresh):
        return -1 / np.log(1 - np.sum(self._p[x >= thresh].values))

    ############
    # plotting #
    ############

    def explore_initial_condition(self, initial_condition_name, var_name, ax=None):
        dead_ends = []
        for k,v in self._forest.items():
            if v.full_name.split('/')[0] == initial_condition_name:
                if len(v.children) == 0:
                    dead_ends.append(v)

        if ax is None:
            fig,ax = plt.subplots()
        #for p in [0,10]:
        #    up,low = tuple(np.percentile(original._data.rolling(time=5, center=True).mean(), [p,100-p], axis=0))
        #    ax.fill_between(range(len(up)), up, low, color=original._color, alpha=0.3)

        for sim in dead_ends:
            x = sim._data[var_name].copy()
            while sim.parent is not None:
                sim = sim.parent
                x = xr.concat((sim._data[var_name], x), dim='time')
            if len(x) == 90:
                linestyle = '-'
            else:
                linestyle = ':'
                
            ax.plot(x, color=self._color, linestyle=linestyle) 