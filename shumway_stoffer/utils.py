import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import datetime as dt
from math import ceil
import pandas as pd


def get_date_from_fractional_year(date_fl: float, freq: int = 52) -> dt.datetime:
    # make sure
    freq = int(freq)
    yyyy = int(date_fl)
    frac_part = date_fl - int(date_fl)
    # just making it throw for other frequencies for now so i catch and can check what is
    # going on
    if freq in [4, 52]:
        week_num = ceil(frac_part * freq)
    else:
        raise RuntimeError(f"Unsupported frequency {freq}")

    # always put it on Sunday (using 0 for %w which the weekday)
    return pd.to_datetime(f"{yyyy} {week_num} 0", format="%Y %W %w")


def load_r_ts(data_rname, r_package: str = "astsa") -> pd.DatetimeIndex:
    robjects.r(f'data({data_rname}, package="astsa")')
    r_data = robjects.r[data_rname]
    r_data_idx = robjects.r["time"](r_data)

    tsp = robjects.r["attr"](r_data, "tsp")

    with localconverter(robjects.default_converter + pandas2ri.converter):
        # Convert R objects to NumPy/Pandas compatible objects
        py_values = robjects.conversion.get_conversion().rpy2py(r_data)
        # py_index = robjects.conversion.get_conversion().rpy2py(r_data_idx)

    freq = int(tsp[2])

    data_index = pd.DatetimeIndex(
        [get_date_from_fractional_year(tt, freq=freq) for tt in r_data_idx]
    )
    return pd.DataFrame({"value": py_values, "index": data_index})
