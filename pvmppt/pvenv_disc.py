import logging
import os

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step as ts

from data_manager import parse_depfie_csv
from pv_array import PVArray


class PVEnv(py_environment.PyEnvironment):
    """The states are a 1 x 3 array, where: state[0] (v_pv) is the output voltage, state[1] (p_pv) is the output power, state[2] (v_pv_delta) is the change applied to the voltage in the previous state.
    The action (v_pv_delta) is a escalar to indicate the next perturbation on the output voltage.
    """

    def __init__(
        self,
        pv_params,
        pv_model_path,
        weather_csv_path,
        discount=1.0,
        seed=None,
        max_episode_steps=None,
        delta_v=0.1,
        v0=None,
        num_discrete_actions=3,
    ):
        """Initializes PVEnv.
        
        Args:
            #TODO
        """
        np.random.seed(seed)

        assert num_discrete_actions > 2
        if num_discrete_actions % 2 == 0:
            num_discrete_actions += 1
        action_min = -(num_discrete_actions // 2)
        action_max = num_discrete_actions // 2

        self.weather_df = parse_depfie_csv(weather_csv_path)
        self.pv_array = PVArray(pv_params, pv_model_path, f_precision=2)

        self._action_spec = BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            name="action",
            minimum=0,
            maximum=num_discrete_actions - 1,
        )
        self._observation_spec = BoundedArraySpec(
            shape=(3,),
            dtype=np.float32,
            name="observation",
            minimum=[0, 0, action_min],
            maximum=[1e4, 1e4, action_max],
        )

        self._discount = discount
        self._max_episode_steps = max_episode_steps or len(self.weather_df)
        self._delta_v = delta_v
        self._num_discrete_actions = num_discrete_actions
        self._reward_normalizer = float(pv_params["vm"]) * float(pv_params["im"])
        self._v0 = v0

        logging.debug(f"Num discrete actions: {num_discrete_actions}")
        logging.debug(f"Action min: {action_min}")
        logging.debug(f"Action max: {action_max}")

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _clip_var(self, value, minimum=-np.inf, maximum=np.inf):
        # if value < minimum:
        #     return minimum
        # if value > maximum:
        #     return maximum
        # return value
        return min(max(value, minimum), maximum)

    def _reset(self):
        v0 = self._v0 or np.random.randint(0, self.pv_array.voc)
        self._state = [v0, 0.0, 0.0]
        self._step_counter = 0
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        # print('Env Step')
        if self._episode_ended:
            return self._reset()

        delta_v = (action - self._num_discrete_actions // 2) * self._delta_v
        pv_voltage = self._clip_var(self._state[0] + delta_v, 0, self.pv_array.voc)

        # weather_idx = np.random.randint(self.weather_df_len)
        irradiance = self.weather_df["Irradiance"].iloc[self._step_counter]
        temperature = self.weather_df["Temperature"].iloc[self._step_counter]

        pv_sim_result = self.pv_array.simulate(pv_voltage, irradiance, temperature)
        pv_voltage = pv_sim_result.voltage
        pv_power = pv_sim_result.power

        self._state = [pv_voltage, pv_power, delta_v]
        reward = (pv_power / self._reward_normalizer) ** 2
        self._step_counter += 1

        if pv_power <= 0:
            # self._episode_ended = True
            self._state[1] = 0
            reward = -0.1

        if self._step_counter >= self._max_episode_steps:
            self._episode_ended = True

        logging.debug(
            f"Idx: {self._step_counter:3}, G: {irradiance:4}, T: {temperature:5.2f}, V: {pv_voltage:5.2f}, P: {pv_power:7.2f}, dV: {delta_v:4.1f}, Rew: {reward:12.6f}"
        )

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.float32), reward, discount=self._discount
            )

    # def get_state(self) -> ts.TimeStep:
    #     return self._current_time_step

    # def set_state(self, time_step: ts.TimeStep):
    #     self._current_time_step = time_step
    #     self._states = time_step.observation


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    pv_array_model_1 = {
        "np": "1",
        "ns": "1",
        "n_cells": "54",
        "voc": "32.9",
        "isc": "8.21",
        "vm": "26.3",
        "im": "7.61",
        "beta_voc": "-0.1230",
        "alpha_isc": "0.0032",
        "bal": "on",
        "tc": "1e-6",
    }
    model_path = os.path.join(".", "pv_array")
    df_path = os.path.join(".", "data", "constant_weather.csv")
    env = PVEnv(
        pv_array_model_1,
        model_path,
        df_path,
        max_episode_steps=100,
        num_discrete_actions=11,
    )
    validate_py_environment(env, episodes=5)

    print("\n\nDONE")
