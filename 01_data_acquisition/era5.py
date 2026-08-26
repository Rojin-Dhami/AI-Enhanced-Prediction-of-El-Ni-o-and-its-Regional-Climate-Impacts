import cdsapi
c = cdsapi.Client()


c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                'sea_surface_temperature',
                'mean_sea_level_pressure',
                'mean_surface_eastward_turbulent_stress',
                'mean_surface_northward_turbulent_stress',
                'top_net_thermal_radiation'  # OLR
            ],
            'year': str(1980),
            'month': [f'{m:02d}' for m in range(1, 13)],
            'time': '00:00',
            'grid': [2.0, 2.0],
            'area': [30, 120, -30, -80],  # N, W, S, E
            'format': 'netcdf'
        },
        f'era5_singlelevels_{1980}.nc'
    )