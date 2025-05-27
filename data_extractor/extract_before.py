for climate in ['c','p']:
    exp = experiment(importlib.import_module(f"experiment_configuration.{climate}1").config)

    naming_d = {
        "project": 'REA_output',
        "product": exp.product_name,
        "institute": 'NCAR',
        "model": 'CESM2',
        "experiment" : f"{exp.initial_conditions_name}-initial",
        "time_frequency": "day",
        "realm": "atmos",
        "variable": "pr-reg-before",
    }

    pr_initial = xr.open_mfdataset(f"/work/bb1152/u290372/REA_output/heat_wEU_JJA/NCAR/CESM2/{naming_d['experiment']}/day/atmos/pr-reg/*/pr-reg_day_CESM2_{naming_d['experiment']}_*.nc", combine='nested', concat_dim='sim')

    for i,sim_name in enumerate(pr_initial.sim.values):
        ens_name = f"ens{str(i+1).zfill(3)}"
        initial_archive = [s for s in exp.initial_conditions if s.split('/')[-1][:4] == sim_name.split('_')[-1] and s.split('.fE.')[0].split('.')[-1] == sim_name.split('_')[0] ][0]
        if climate == 'p':
            initial_before_archive = initial_archive.split('/branch/')[0]
            initial_year = initial_archive.split('/')[-1][:4]
        else:
            initial_before_archive = '/'.join(initial_archive.split('/')[:-1])
            initial_year = int(initial_archive.split('/')[-1][:4])
        with xr.open_mfdataset(f"{initial_before_archive}/atm/hist/*h1.{initial_year}*") as nc:
            nc = shift_lon(nc)
            v = nc['PRECC'] + nc['PRECL'] 
            v *= 24*60*60
            v = v.loc[:f"{initial_year}-{exp.start_date_in_year}"]

            # regional mask
            regional_mask = create_or_load_regional_mask(
                regional_mask_file = f"/work/bb1152/u290372/GKLT/regions/wEU.nc",
                slice_lat=slice(44,55), 
                slice_lon=slice(-4,12),
                )

            v = regional_average(v, regional_mask)
            v = v.expand_dims({'sim':1})
            v = v.assign_coords(sim = [sim_name])
            v = v.assign_coords(time=xr.date_range(f"{exp.initial_condition_fake_year}-01-01", periods=150))
        
            out_dir = '/'.join([exp.dir_work] + [v for k,v in naming_d.items()] + [ens_name])
            out_file_name = f"{out_dir}/{naming_d['variable']}_{naming_d['time_frequency']}_CESM2_{naming_d['experiment']}_{ens_name}_{exp.initial_condition_fake_year}.nc"
            
            ds = xr.Dataset({'pr':v})
            ds.attrs = nc.attrs
            ds.attrs['simulation_name'] = sim_name
            ds.attrs['initial_condition'] = exp.initial_conditions[i]
            ds.attrs['initial_condition_year'] = exp.initial_conditions[i].split('/')[-1].split('-')[0]
            ds.attrs['compset'] = exp.compset
            ds.attrs['readme'] = exp.git_repo
            ds['time'].attrs['comment'] = f'This simulation represents the climate state of {exp.initial_conditions_name}. The year in the time axis can be ignored.'

            os.makedirs(out_dir, exist_ok=True)
            ds.to_netcdf(out_file_name)