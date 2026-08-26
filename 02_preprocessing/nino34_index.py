import pandas as pd

nino34 = pd.read_csv(
    'https://psl.noaa.gov/data/correlation/nina34.data',
    sep=r'\s+', skiprows=1, header=None
)

mei = pd.read_csv(
    'https://psl.noaa.gov/enso/mei/data/meiv2.data',
    sep=r'\s+', skiprows=1, header=None
)