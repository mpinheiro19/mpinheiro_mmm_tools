import datetime as dt

import numpy as np
import pandas as pd
from numpy.random import default_rng

# Setting random seed value
SEED = 42

# Set the number of distinct features
N = 10

# Instanciate numpy's random generator class
rng = default_rng(SEED)

# Set the time range
start_date = dt.datetime(2022, 1, 1)
end_date = dt.datetime(2022, 12, 31)

if __name__ == "__main__":
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Create a random time series with N distinct features
    data = rng.integers(
        low=0, high=65000, size=(len(date_range), N), endpoint=True
    )

    # Create a Pandas DataFrame with the time series data
    df = pd.DataFrame(
        data, index=date_range, columns=["ft_" + str(i + 1) for i in range(N)]
    )

    # df.to_csv("./data/raw/fake_series.csv")
