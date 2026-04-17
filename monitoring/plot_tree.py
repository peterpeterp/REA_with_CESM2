import sys,os
import numpy as np
import distinctipy

import matplotlib.pyplot as plt

# get some paths and general settings
sys.path.append('/home/u/u290372/projects/REA_with_CESM2')
from ensembles.ensemble_GKLT import ensemble_GKLT
from ensembles.ensemble_GKLT_legacy import ensemble_GKLT_legacy


def last_finished_step(exp):
    for step in range(exp.n_steps, -1, -1):
        # check if previous step has been started
        previous_todo_csv = f"{exp.dir_work}/GKLT/{exp.experiment_name}/book_keeping/step{step-1}_evaluation.csv"
        if os.path.isfile(previous_todo_csv):
            break
    return step

def plot_tree(exp,end_step, color_seed=1):
    colors = distinctipy.get_colors(exp.n_members, rng=color_seed)
    if exp.ensemble_type == 'rea':
        ens = ensemble_GKLT(exp)
    elif exp.ensemble_type == 'rea_legacy':
        ens = ensemble_GKLT_legacy(exp)
    else:
        assert False, 'problem'
    ens.get_sim_names(end_step=end_step)
    ens.build_forest()
    fig,ax = plt.subplots()
    x = np.zeros([end_step+1])
    y = np.arange(0,end_step+1,1)
    i = 0
    initials = {}
    for sim_name,sim in ens._forest.items():
        sim_name = '.'.join(sim.name.split('.')[1:])
        initial_name = sim_name.split('.')[0]
        if len(sim_name.split('.')) == end_step + 1:
            if i == 0:
                prev_names = sim_name.split('.')
            else:
                for j, name in enumerate(sim_name.split('.')):
                    if prev_names[j] != name:
                        x[j] += int(name) - int(prev_names[j])
                        prev_names[j] = name
            if initial_name not in initials.keys():
                initials[initial_name] = []
            initials[initial_name].append(x.copy())
            i += 1

    for i_color, initial_name in enumerate(initials.keys()): 
        for x in initials[initial_name]:
            ax.plot(x, y, color=colors[i_color])

    ax.set_xlabel('initial condition')
    ax.set_xticks(np.arange(1,128,12)-1)
    ax.set_yticks(range(0,end_step+1))
    ax.set_ylabel('simulation step')