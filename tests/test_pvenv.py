import os
import pytest

from ..core.pvenv import PVEnv
from ..core.pv_array import PVArray
from ..core.data_manager import parse_depfie_csv


pv_array_params = {
    "Npar": "1",
    "Nser": "1",
    "Ncell": "54",
    "Voc": "32.9",
    "Isc": "8.21",
    "Vm": "26.3",
    "Im": "7.61",
    "beta_Voc_pc": "-0.1230",
    "alpha_Isc_pc": "0.0032",
    "BAL": "on",
    "Tc": "1e-6",
}

pvarray = PVArray(pv_array_params, float_precision=8)

weather_df = parse_depfie_csv(os.path.join(os.getcwd(), "data", "toy_weather.csv"))


def test_empty_env():
    with pytest.raises(NotImplementedError):
        pvenv = PVEnv(pvarray, weather_df)
