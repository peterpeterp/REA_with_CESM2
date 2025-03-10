from ensembles.ensemble import * 

class ensemble_initial(ensemble):
    def __init__(self, exp):
        self._name = f"{exp.region_of_interest}.{exp.observable_of_interest}.{exp.start_date_in_year}.{exp.n_days}x{exp.n_steps}.{exp.initial_conditions_name}.initial"
        super().__init__(exp)
        h1_dict = self.get_h_dict('atm', 'h1')
        self._sim_names = sorted(h1_dict.keys())
        self._trajectories = {}

        self._weight = xr.DataArray(1, dims=['sim','step'], coords=dict(sim=self._sim_names,step=np.arange(0,self._exp.n_steps,1,'int')))
        self._weight_daily = xr.DataArray(
            1.,
            dims = ['sim','day'],
            coords = dict(sim=self._sim_names, day=np.arange(0,self._exp.n_days * self._exp.n_steps,1,'int'))
        )


    def get_h_dict(self, realm, h_identifier):
        h_dict = {}
        for initial in sorted(self._exp.initial_conditions):
            sim_path = ''
            for p in initial.split('/'):
                if p == 'branch' or '_to_' in p:
                    break
                else:
                    sim_path += f"{p}/"
            sim = sim_path.split('/')[-2]
            year = int(initial.split('/')[-1].split('-')[0])
            h_dict[f"{sim}_{year}"] = glob.glob(f"{sim_path}/{realm}/hist/*.{h_identifier}.*{year}*.nc") 
        return h_dict


    def extract_1D(self, h_identifier, realm, extractor, var_name=None):
        if var_name is None:
            var_name = extractor.__name__
        file_name = f"{self._dir}/data/{var_name}.nc"
        if os.path.isfile(file_name):
            self._trajectories[var_name] = xr.open_dataset(file_name)[var_name].load()
        else:
            h_dict = self.get_h_dict(realm, h_identifier)
            self._trajectories[var_name] = xr.DataArray(dims=['sim','day'], coords=dict(sim=self._sim_names, day=np.arange(self._exp.n_steps * self._exp.n_days)))
            for sim_name in self._sim_names:
                year = sim_name.split('_')[-1].zfill(4)
                return extractor(h_dict[sim_name], year)
                self._trajectories[var_name].loc[sim_name] = extractor(h_dict[sim_name], year).values
            xr.Dataset({var_name:self._trajectories[var_name]}).to_netcdf(file_name)

    def extract_3D(self, h_identifier, extractor, overwrite=False, var_name=None):
        if var_name is None:
            var_name = extractor.__name__
        file_name = f"{self._dir}/data/{var_name}.nc"
        if os.path.isfile(file_name) and overwrite == False:
            self._trajectories[var_name] = xr.open_dataset(file_name)[var_name]
        else:
            h_dict = self.get_h_dict(h_identifier)
            trajs = []
            for sim_name,fl in h_dict.items():
                with xr.open_dataset(fl) as nc:
                    year = sim_name.split('_')[-1].zfill(4)
                    jja_times = nc.time.loc[f'{year}-{self._exp.start_date_in_year}':][1:1+self._exp.n_days*self._exp.n_steps]
                    x = extractor(nc.sel(time=jja_times))
                    x = x.assign_coords(time=np.arange(x.shape[0]))
                    x = x.rename(time='day')
                    x = x.assign_coords(sim=sim_name)
                    trajs.append(x)
            self._trajectories[var_name] = xr.concat(trajs, dim='sim')
            xr.Dataset({var_name:self._trajectories[var_name]}).to_netcdf(file_name)

