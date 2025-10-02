from ensembles.ensemble import * 

import multiprocessing
import pandas as pd

import matplotlib.pyplot as plt

from anytree import NodeMixin, Node, RenderTree, AsciiStyle, findall

def get_weight_for_selection(sim_names, exp):
    weight = xr.DataArray(dims=['sim','step'], coords=dict(sim=sim_names,step=np.arange(0,exp.n_steps,1,'int')))
    weight_daily = xr.DataArray(
        dims=['sim','step','time'], 
        coords=dict(sim=sim_names,step=np.arange(0,exp.n_steps,1,'int'),time=np.arange(0,exp.n_days,1,'int'))
    )
    sim_names = np.array([s.split('/') for s in sim_names])

    for step in weight.step.values:
        for v in np.unique(sim_names[:,step]):
            same = (sim_names[:,step] == v)
            weight[same, step] = 1 / same.sum()
            weight_daily[same, step] = 1 / same.sum()

    return weight.mean('step')


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
        
        # Ris
        self._mean_scores = np.array([
            pd.read_table(f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step}_evaluation.csv", sep=',')['score'].values.mean()
            for step in range(self._exp.n_steps-1)
        ])

    ###################################
    # Get data structure of ensemble  #
    # build a forest                  #
    ###################################

    def get_sim_names(self, overwrite=False):
        file_name = f"{self._exp.dir_out}/sim_names.txt"
        if os.path.isfile(file_name) == False or overwrite:
            todo_tables = {}
            steps = np.arange(0,self._exp.n_steps,1,'int')
            for step in steps:
                todo_tables[step] = pd.read_table(f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step}.csv", sep=',')
                todo_tables[step]['parent_identifier'] = [p.split('/')[-1] for p in todo_tables[step]['parent_path']]


            self._sim_names = []
            for i in todo_tables[step].index:
                step_names = [ todo_tables[steps[-1]].loc[i, 'case_identifier'].split('_')[-1] ]
                parent_name = todo_tables[steps[-1]].loc[i, 'parent_identifier']
                for step in steps[::-1][1:]:
                    step_names += [ todo_tables[step].loc[todo_tables[step].case_identifier == parent_name, 'case_identifier'].values[0].split('_')[-1] ]
                    parent_name = todo_tables[step].loc[todo_tables[step].case_identifier == parent_name, 'parent_identifier'].values[0]
                sim_name = '.'.join([ self._exp.experiment_identifier ] + step_names[::-1])
                self._sim_names.append(sim_name)
            self._sim_names = np.array(self._sim_names)
            with open(file_name, 'w') as fl:
                fl.write(';'.join(self._sim_names))
        else:
            self._sim_names = open(file_name, 'r').read().split(';')

    def get_weights_uniqueness(self):
        '''
        weights based on uniqueness
        '''

        self._uniq_stepwise = xr.DataArray(dims=['sim','step'], coords=dict(sim=self._sim_names,step=np.arange(0,self._exp.n_steps,1,'int')))
        uniq_daily = xr.DataArray(
            dims=['sim','step','time'], 
            coords=dict(sim=self._sim_names,step=np.arange(0,self._exp.n_steps,1,'int'),time=np.arange(0,self._exp.n_days,1,'int'))
        )
        sims = np.array([s.split('.')[1:] for s in self._sim_names])
        for step in self._uniq_stepwise.step.values:
            for v in np.unique(sims[:,step]):
                same = (sims[:,step] == v)
                self._uniq_stepwise[same, step] = 1 / same.sum()
                uniq_daily[same, step] = 1 / same.sum()

        self._uniq_seas = self._uniq_stepwise.mean('step')
        self._uniq_daily = xr.DataArray(
                    uniq_daily.values.reshape((self._exp.n_members, self._exp.n_steps * self._exp.n_days)),
                    dims = ['sim','time'],
                    coords = dict(sim=self._uniq_stepwise.sim, time=np.arange(0,self._exp.n_days * self._exp.n_steps,1,'int'))
                )

    ##################
    # First analysis #
    ##################

    def evaluate_weights_and_probabilities(self, obs):
        self._obs = obs
        self._abs = self._obs.mean('time')

        x = xr.DataArray(
            self._obs.values.reshape((-1, self._exp.n_steps, self._exp.n_days)),
            dims=['sim','step','time'],
            coords=dict(sim=self._obs.sim, step=np.arange(0,self._exp.n_steps,1,'int'), time=np.arange(0,self._exp.n_days,1,'int'))
        )

        if self._exp.ensemble_type == 'rea':
            self._time_aggregation_over_each_step = x.mean('time')
        elif self._exp.ensemble_type == 'rea_legacy':
            self._time_aggregation_over_each_step = x.sum('time')

        self._scores = np.exp(self._exp.k * self._time_aggregation_over_each_step)

        self._weight_from_algo = self._obs.mean('time').copy() * np.nan
        self._weight_from_algo[:] = np.array(
            [
                np.product(self._mean_scores) / (np.product(self._scores[i,:-1]) * self._exp.n_members)
                for i in range(self._exp.n_members)
            ]
        )

        self._prob = self._obs.mean('time').copy() * np.nan
        self._prob[:] = np.array([np.sum((self._abs >= a).astype(float) * self._weight_from_algo) for a in self._abs]) 

    def ra(self, x, thresh):
        return -1 / np.log(1 - np.sum(self._weight_from_algo[x >= thresh].values))

    ############
    # plotting #
    ############

    def explore_initial_condition(self, initial_condition_name, var_name, ax=None, max_length=90):
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
            if len(x) == max_length:
                linestyle = '-'
                color = 'r'
            else:
                linestyle = '-'
                color='gray'

            ax.plot(x, color=color, linestyle=linestyle, linewidth=0.5) 