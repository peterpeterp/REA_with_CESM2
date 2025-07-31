import os,sys,subprocess,glob,importlib
from datetime import datetime
import xarray as xr
import numpy as np
import pandas as pd

# get some paths and general settings
from settings import *


class launch_handler():
    def __init__(self, experiment, verbose=True, dry_run=True, relaunch_cases_which_are_unclear=False, relaunch_after_completion=False):
        self._exp = experiment
        self._verbose = verbose
        self._dry_run = dry_run
        self._relaunch_after_completion = relaunch_after_completion
        self._relaunch_cases_which_are_unclear = relaunch_cases_which_are_unclear

        # create dirs
        for path in [
            self._exp.dir_archive,
            self._exp.dir_run,
            self._exp.dir_scripts,
            self._exp.dir_work,
            ]:
            path += f"/GKLT/{self._exp.experiment_name}"
            if os.path.isdir(path) == False:
                self.run(f"mkdir -p {path}")

        for path in [
            f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping",
            f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/logs_of_fails",
        ]:
            if os.path.isdir(path) == False:
                self.run(f"mkdir -p {path}")

        self.run('module unload git')

    def check_status_of_todo(self, todo):
        '''
        Parameters:
            todo (pd.Dataframe row): a row from a todo-table

        Returns:
            status (str)
        '''
        if os.path.isdir(f"{self._exp.dir_archive}/{todo.loc['case_path']}/{todo.loc['case_identifier']}/rest"):
            return 'done'
        # check squeue
        my_jobs = subprocess.check_output(["squeue", '--me', '--format="%.18i %.70j %.8T %.10M"']).decode().split('\n')
        if len([l for l in my_jobs if todo.loc['case_identifier'] in l]) > 0:
            return 'running'
        # check if job has been launched previously
        if os.path.isfile(f"{self._exp.dir_scripts}/{todo.loc['case_path']}/{todo.loc['case_identifier']}/CaseStatus"):
            return 'unclear'
        # if none of the previous conditions met the job can be launched
        return 'not launched'

    def generate_launch_command(self, todo):
        '''
        Parameters:
            todo (pd.Dataframe row): a row from a todo-table

        Returns:
            command (str)
        '''
        return f"python {self._exp.launching_script} " + " ".join([f"--{k} {v}" for k,v in todo.to_dict().items() if v != ""])

    def treat_todo(self, todo):
        status = self.check_status_of_todo(todo)
        if status == 'not launched':
            self.run(self.generate_launch_command(todo))

    # run and print or just print or just run depending on command line parameters
    def run(self, command):
        command = command.replace('\n',' ')
        if self._verbose:
            print(f"\n---- subprocess call:\n{command}")
        if self._dry_run == False:
            return subprocess.run(command, shell=True, check=True)#, stdout=subprocess.PIPE)


    def get_logfiles_of_runs_where_the_status_is_unclear(self):
        # go backwards from last step
        for step in range(self._exp.n_steps, -1, -1):
            # check if previous step has been started
            previous_todo_csv = f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step-1}.csv"
            if os.path.isfile(previous_todo_csv):
                break
        # check if previous step is completed
        previous_todos = pd.read_csv(previous_todo_csv, index_col=0, keep_default_na=False)
        status = np.array(['']*self._exp.n_members, dtype='<U20')
        previous_todos=previous_todos.reindex(columns=[previous_todos.columns[-1]] + list(previous_todos.columns[:-1]))
        for i in previous_todos.index:
            status[i] = self.check_status_of_todo(previous_todos.loc[i])

        log_files = []
        for i in np.where(status == 'unclear')[0]:
            case_status = f"{previous_todos.iloc[i].loc['case_path']}/{previous_todos.iloc[i].loc['case_identifier']}/CaseStatus"
            case_status = f"{self._exp.dir_scripts}/{case_status}"
            if os.path.isfile(case_status):
                for line in open(case_status, 'r').read().split('\n'):
                    if "See log file for details: " in line:
                        log_files.append(line.split("See log file for details: ")[-1])
        return log_files       


    def clean_run_directories_of_step_X(self, step):
        # identify case that is used as source for precompiled copy
        todo_table_step_0 = self.prepare_todos_for_step_0()
        precompiled_path_used_later_on = todo_table_step_0.loc[1, 'precompiled_path']
        run_directories_of_case = sorted(glob.glob(f"{dir_run}/GKLT/{self._exp.experiment_name}/step{step}/*/run"))
        for run_directory_of_case in run_directories_of_case:
            if run_directory_of_case != f"{dir_run}/{precompiled_path_used_later_on}/run":
                self.run(f"rm -rf {run_directory_of_case}")

    def delete_restart_files_of_step_X(self, step):
        # identify case that is used as source for precompiled copy
        rest_directories = sorted(glob.glob(f"{self._exp.dir_archive}/GKLT/{self._exp.experiment_name}/step{step}/*/rest"))
        for rest_directory in rest_directories:
            self.run(f"rm -rf {rest_directory}")

    def prepare_todos_for_step_0(self):
        ##############
        # first step #
        ##############

        # make list of simulations to launch
        l = []
        for member in range(self._exp.n_members):
            d = self._exp.launch_template.copy()
            d['parent_path'] = self._exp.initial_conditions[member]
            d['case_path'] = f"GKLT/{self._exp.experiment_name}/step0"
            d['perturbation_seed'] = 1 + member*10 + self._exp.seed
            d['case_identifier'] = f"{self._exp.experiment_identifier}_{str(member).zfill(3)}"
            l.append(d)

        # convert to table
        todo_table = pd.DataFrame(l)

        # use compilation of first launched simulation
        precompiled_name = f"GKLT/{self._exp.experiment_name}/step0/{todo_table.loc[0,'case_identifier']}"
        todo_table['precompiled_path'] = precompiled_name
        todo_table.loc[0, 'precompiled_path'] = ""
        return todo_table

    def evaluate_previous_step(self, step):
        ##########################
        # evaluate previous step #
        ##########################
        previous_todos = pd.read_csv(f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step-1}.csv", index_col=0, keep_default_na=False)

        # get identifiers from previous step
        evaluation_table = previous_todos.loc[:,["case_identifier"]]
        # extract observables
        for i in evaluation_table.index:
            _archive = f"{self._exp.dir_archive}/{previous_todos.iloc[i].loc['case_path']}/{previous_todos.iloc[i].loc['case_identifier']}"
            obs = self._exp.get_main_observable(_archive)
            evaluation_table.loc[i,'observable mean'] = obs
            # calculate scores
            evaluation_table.loc[i, 'score'] = self._exp.calc_score(obs)
        # calculate weights
        evaluation_table['weight'] = evaluation_table['score'] / np.mean(evaluation_table['score'])
        # normalize weights
        # I think this is useless
        # evaluation_table['weight'] = evaluation_table['weight'] / evaluation_table['weight'].mean()
        return evaluation_table

    def create_list_of_clones(self, step, evaluation_table):
        # set random seed
        np.random.seed(self._exp.seed + step)
        # get number of copies
        evaluation_table['copies'] = ( evaluation_table['weight'] + np.random.uniform( low=0 , high=1 , size=self._exp.n_members )).astype(int)

        # store evaluation
        evaluation_table.to_csv(f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step-1}_evaluation.csv")

        # make a todo-list with clones of indices of the previous_todos
        clones_of = np.zeros( np.sum(evaluation_table['copies'].values) , dtype=int )
        indice_start = 0
        for i in range(self._exp.n_members):
            clones_of[indice_start:indice_start + evaluation_table['copies'][i]] = i
            indice_start += evaluation_table['copies'][i]

        # modify such that the number of members remains unchanged
        delta_N = np.sum( evaluation_table['copies'] ) - self._exp.n_members

        if delta_N >= 0 :
            '''
            select delta_N clones and remove those
            '''
            to_kill = np.random.choice( clones_of , size=delta_N , replace=False )
            for x in to_kill:
                clones_of = np.delete(clones_of, np.where(clones_of==x)[0][0])
        else:
            '''
            add some additional clones from the list 
            '''
            additional = np.random.choice(clones_of, size=-delta_N , replace=False )
            clones_of = np.append(clones_of, additional)


        return clones_of

    def prepare_todos_for_step_X(self, step):
        ##############################################
        # create a list of clone-todos for this step #
        ##############################################

        previous_todos = pd.read_csv(f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step-1}.csv", index_col=0, keep_default_na=False)

        evaluation_table = self.evaluate_previous_step(step)

        clones_of = self.create_list_of_clones(step, evaluation_table)

        # make list of simulations to launch
        l = []
        member = 0
        for i_p in np.unique(clones_of):
            for c in range(np.sum(clones_of == i_p)):
                d = self._exp.launch_template.copy()
                d['case_path'] = f"GKLT/{self._exp.experiment_name}/step{step}"
                d['parent_path'] = f"{self._exp.dir_archive}/{previous_todos.loc[i_p, 'case_path']}/{previous_todos.loc[i_p, 'case_identifier']}"
                d['perturbation_seed'] = 1 + step + i_p * 10 + c + self._exp.seed
                d['case_identifier'] = f"{self._exp.experiment_identifier}_{str(member).zfill(3)}"
                l.append(d)
                member += 1

        # convert to table
        todo_table = pd.DataFrame(l)
        # reuse the same compilation
        todo_table['precompiled_path'] = previous_todos.loc[1, 'precompiled_path']
        return todo_table

    def do_what_has_to_be_done(self):


        # go backwards from last step
        for step in range(self._exp.n_steps, -1, -1):
            # check if previous step has been started
            previous_evaluation_csv = f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step-1}_evaluation.csv"
            if os.path.isfile(previous_evaluation_csv):
                break
        print(f"current step {step}")

        if step == 0:
            todo_table = self.prepare_todos_for_step_0()
            # launch simulations
            for i in todo_table.index:
                self.treat_todo(todo_table.loc[i])
            todo_table.to_csv(f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step}.csv")
            print(f"working on step {step}")

        else:
            # check if previous step is completed
            previous_todos = pd.read_csv(previous_todo_csv, index_col=0, keep_default_na=False)
            status_l = np.array(['']*self._exp.n_members, dtype='<U20')
            previous_todos=previous_todos.reindex(columns=[previous_todos.columns[-1]] + list(previous_todos.columns[:-1]))
            for i in previous_todos.index:
                status_l[i] = self.check_status_of_todo(previous_todos.loc[i])

            if np.all(status_l == 'done'):
                print(f"step {step} is done")
                print(f"cleaning run directories of {step -1}")
                self.clean_run_directories_of_step_X(step - 1)
                print(f"deleting restart files of {step - 2}")
                self.delete_restart_files_of_step_X(step - 2)
                print(f"need to do step {step}")
                todo_table = self.prepare_todos_for_step_X(step)
                todo_table.to_csv(f"{self._exp.dir_work}/GKLT/{self._exp.experiment_name}/book_keeping/step{step}.csv")

                # launch simulations
                for i in todo_table.index:
                    self.treat_todo(todo_table.loc[i])
                print(f"working on step {step}")

            elif np.any(status_l == 'not launched'):                
                print(f"launching remaining simulations for step {step}")
                print(f"to be launched: {', '.join(previous_todos.loc[status_l == 'not launched', 'case_identifier'].values)}")
                for i in previous_todos.index:
                    self.treat_todo(previous_todos.loc[i])

            elif np.all((status_l == 'done') | (status_l == 'running')):
                print(f"waiting for step {step}")
                print(f"still running: {', '.join(previous_todos.loc[status_l == 'running', 'case_identifier'].values)}")
                pass

            else:
                print('-'*10 + ' overview ' + '-'*10)
                for status_type in ['running', 'unclear', 'not launched']:
                    print(f"{status_type}: {', '.join(previous_todos.loc[status_l == status_type, 'case_identifier'].values)}")

                
                print('-'*10 + ' logs ' + '-'*10)
                for status_type in ['unclear']:
                    log_files = []
                    for i in np.where(status_l == status_type)[0]:
                        case_status = f"{previous_todos.iloc[i].loc['case_path']}/{previous_todos.iloc[i].loc['case_identifier']}/CaseStatus"
                        print(case_status)
                        log = open(f"{self._exp.dir_scripts}/{case_status}", "r").read().split("\n")
                        important_lines = [l for l in log if "See log file for details: /scratch/u/u290372/" in l]
                        if len(important_lines) >= 1:
                            scratch_log = important_lines[-1].split(' ')[-1]
                            self.run(f"cp {scratch_log} {self._exp.dir_out}/logs_of_fails/{scratch_log.split('/')[-1]}")

                print('-'*10 + ' launch commands ' + '-'*10)
                for status_type in ['unclear', 'not launched']:
                    for i in np.where(status_l == status_type)[0]:
                        print(self.generate_launch_command(previous_todos.iloc[i]))
                        if self._relaunch_cases_which_are_unclear:
                            self.run(self.generate_launch_command(previous_todos.iloc[i]) + ' --overwrite')


                if self._relaunch_cases_which_are_unclear == False:
                    print('needs a fix')
                    exit(0)          

        if self._relaunch_after_completion:
            self.resubmit_after_completion_of_previous_runs(step + 1)


    def resubmit_after_completion_of_previous_runs(self, step):
        if step <= self._exp.n_steps:
            print(f"resubmiting the same launcher script for step {step}")
            my_jobs = subprocess.check_output(["squeue", '--me', '--format="%.18i %.70j %.8T %.10M"']).decode().split('\n')
            my_jobs_associated_to_this_experiment = [j for j in my_jobs if self._exp.experiment_identifier+'_' in j]
            job_ids_to_wait_for = [j.split()[1] for j in my_jobs_associated_to_this_experiment]

            # this did not work because some jobs finish before the next one is launched ?!?
            #SBATCH --job-name=launch_{self._exp.experiment_identifier}_step{step+1}

            new_slurm_job = sbatch_job_header
            new_slurm_job += f"#SBATCH --job-name={self._exp.experiment_identifier}_step{step}\n"
            new_slurm_job += f"#SBATCH --account={self._exp.dkrz_project_for_accounting}\n"
            new_slurm_job += f"#SBATCH --begin=now+10minute\n"

            if len(job_ids_to_wait_for) > 0:
                new_slurm_job += f"#SBATCH --dependency=afterok:{','.join(job_ids_to_wait_for)}\n"

            new_slurm_job += sbatch_modules

            new_slurm_job += f"python main_launcher.py --experiment {self._exp.experiment_identifier}"
            for cmd_line_argument in ['verbose','dry_run','relaunch_cases_which_are_unclear', 'relaunch_after_completion']:
                if self.__dict__[f"_{cmd_line_argument}"]:
                    new_slurm_job += f" --{cmd_line_argument}"

            with open(f"slurm_job_files/job_{self._exp.experiment_name}", 'w') as job_file:
                job_file.write(new_slurm_job)

            self.run(f"sbatch slurm_job_files/job_{self._exp.experiment_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--dry_run", action='store_true')
    parser.add_argument("--relaunch_cases_which_are_unclear", action='store_true')
    parser.add_argument("--relaunch_after_completion", action='store_true')
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
        relaunch_cases_which_are_unclear=command_line_arguments.relaunch_cases_which_are_unclear, 
        relaunch_after_completion=command_line_arguments.relaunch_after_completion
        )

    launcher.do_what_has_to_be_done()

