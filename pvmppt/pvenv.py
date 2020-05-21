import logging
import os

import numpy as np
import pandas as pd

from tf_agents.environments import py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step as ts

from typing import Optional

# from .data_manager import parse_depfie_csv
from .pv_array import PVArray


class PVEnv(py_environment.PyEnvironment):
    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        discount: float = 1.0,
        max_episode_steps: Optional[int] = None,
        v0: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.pvarray = pvarray
        self.weather_df = weather_df

        self._max_episode_steps = max_episode_steps or len(self.weather_df)
        self._discount = discount
        self._reward_normalizer = float(pvarray.model_params["Vm"]) * float(
            pvarray.model_params["Im"]
        )
        self._v0 = v0

        if seed:
            np.random.seed(seed)

        self._action_spec = self._set_action_spec()
        self._observation_spec = self._set_observation_spec()

    def action_spec(self) -> BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> BoundedArraySpec:
        return self._observation_spec

    def _set_action_spec(self) -> BoundedArraySpec:
        raise NotImplementedError

    def _set_observation_spec(self) -> BoundedArraySpec:
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError


class PVEnvDisc(PVEnv):
    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        discount: float = 1.0,
        max_episode_steps: Optional[int] = None,
        v0: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        pass


if __name__ == "__main__":
    pass
#     logging.basicConfig(level=logging.DEBUG)

#     pv_array_model_1 = {
#         "np": "1",
#         "ns": "1",
#         "n_cells": "54",
#         "voc": "32.9",
#         "isc": "8.21",
#         "vm": "26.3",
#         "im": "7.61",
#         "beta_voc": "-0.1230",
#         "alpha_isc": "0.0032",
#         "bal": "on",
#         "tc": "1e-6",
#     }
#     model_path = os.path.join(".", "pv_array")
#     df_path = os.path.join(".", "data", "constant_weather.csv")
#     env = PVEnv(
#         pv_array_model_1,
#         model_path,
#         df_path,
#         max_episode_steps=100,
#         num_discrete_actions=11,
#     )
#     validate_py_environment(env, episodes=5)

#     print("\n\nDONE")
