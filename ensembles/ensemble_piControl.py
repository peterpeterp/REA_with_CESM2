from ensembles.ensemble import * 

class ensemble_piControl(ensemble):
    def __init__(self, exp):
        self._name = f"{exp.region_of_interest}.{exp.observable_of_interest}.{exp.start_date_in_year}.{exp.n_days}x{exp.n_steps}.piControl"
        super().__init__(exp)
        self._trajectories = {}
        h_dict = get_h_dict('h1')
        self._sim_names = sorted(h_dict.keys())

    def get_h_dict(self, h_identifier):
        h_dict = {}
        for h_file in sorted(glob.glob(f"{exp.dir_archive}/b.e215.B1850cmip6.f09_g17.001.fE.0500.ens000/atm/hist/*{h_identifier}*")):
            year = int(h_file.split('.')[-2].split('-')[0])
            h_dict[f"{year}"] = h_file
        return h_dict
        

    def extract_1D(self, var_name, h_identifier, extractor):
        file_name = f"{self._dir}/{extractor.__name__}.nc"
        if os.path.isfile(file_name):
            self._trajectories[var_name] = xr.open_dataset(file_name)[extractor.__name__].load()
        else:
            h_dict = get_h_dict('h1')
            self._trajectories[var_name] = xr.DataArray(dims=['sim','day'], coords=dict(sim=self._sim_names, day=np.arange(self._exp.n_steps * self._exp.n_days)))
            for sim_name in self._sim_names:
                with xr.open_dataset(h_dict[sim_name]) as nc:
                    year = sim_name.split('_')[-1].zfill(4)
                    jja_times = nc.time.loc[f'{year}-{self._exp.start_date_in_year}':][1:1+self._exp.n_days*self._exp.n_steps]
                    self._trajectories[var_name].loc[sim_name] = extractor(nc.sel(time=jja_times)).values
            xr.Dataset({extractor.__name__:self._trajectories[var_name]}).to_netcdf(file_name)

