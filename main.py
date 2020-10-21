from src.pv_array import PVArray
import os
import matplotlib.pyplot as plt
import numpy as np

from src.utils import read_weather_csv
from src.pv_env import PVEnv, PVEnvDiscrete

PV_PARAMS_PATH = os.path.join("parameters", "pvarray_01.json")
WEATHER_PATH = os.path.join("data", "weather_real_01.csv")

if __name__ == "__main__":
    pvenv = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,
        WEATHER_PATH,
        discount=0.99,
        max_episode_steps=100,
        v_delta=0.1,
    )
