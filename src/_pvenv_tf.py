import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step as ts

from src.pv_array import PVArray
from src.utils import clip_num, read_weather_csv

REWARD_NEG_POWER = -200


class PVEnv(py_environment.PyEnvironment):
    """
    PV Environment abstract class for solving the MPPT by reinforcement learning

    Parameters:
        - pvarray: the pvarray object
        - weather_df: a pandas dataframe object containing weather readings
        - discount: discount future rewards
            - 0: only account inmmediate reward
            - 1: all rewards weighs the same
        - max_episode_steps: maximum number of steps in the episode
            - None: the episode last until the dataframe is exhausted
        - v0: initial load voltage
        - seed: for reproducibility
        - early_end: whether a negative output finishes the episode
    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        discount: float = 1.0,
        max_episode_steps: Optional[int] = None,
        v0: Optional[float] = None,
        seed: Optional[int] = None,
        early_end: bool = False,
    ) -> None:
        self.pvarray = pvarray
        self.weather_df = weather_df

        self._max_episode_steps = max_episode_steps or len(weather_df) - 1
        self._max_episode_steps = min(self._max_episode_steps, len(weather_df))
        self._discount = discount
        self._reward_normalizer = float(pvarray.model_params["Vm"]) * float(
            pvarray.model_params["Im"]
        )
        self._v0 = v0
        self._early_end = early_end

        if seed:
            np.random.seed(seed)

        self._step_counter = 0
        self._action_spec = None
        self._observation_spec = None

        logging.info(f"PV Environment")
        logging.info(f"Max episodes: {self._max_episode_steps}")
        logging.info(f"Discount: {self._discount}")
        logging.info(f"Reward normalizer {self._reward_normalizer}")
        logging.info(f"Initial V: {self._v0}")
        logging.info(f"Early end: {self._early_end}")

    def action_spec(self) -> BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> BoundedArraySpec:
        return self._observation_spec

    def _reset(self) -> ts.StepType:
        raise NotImplementedError

    def _get_delta_v(self, action: float) -> float:
        raise NotImplementedError

    def _step(self, action) -> ts.StepType:
        raise NotImplementedError


class PVEnvDiscV0(PVEnv):
    """
    PV discrete environment with availability of weather observations.

    Available actions:
        0: decrement v_delta
        1: do nothing
        2: increment v_delta

    Observations:
        [load voltage, irradiance, cell temperature]

    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        v_delta: float,
        discount: float = 1.0,
        max_episode_steps: Optional[int] = None,
        v0: Optional[float] = None,
        seed: Optional[int] = None,
        early_end: bool = True,
    ) -> None:
        super().__init__(
            pvarray,
            weather_df,
            discount,
            max_episode_steps,
            v0,
            seed,
            early_end,
        )

        self._v_delta = v_delta

        self._action_spec = BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            name="action",
            minimum=0,
            maximum=2,
        )
        self._observation_spec = BoundedArraySpec(
            shape=(3,),
            dtype=np.float32,
            name="observation",
            minimum=[0, 0, 0],
            maximum=[self.pvarray.voc, 1200, 50],
        )

        logging.info(f"Action spec: {self.action_spec()}")
        logging.info(f"Obs space: {self.observation_spec()}")
        logging.info(f"Delta V: {self._v_delta}")

    def _get_delta_v(self, action: float) -> float:
        if action == 0:
            return -self._v_delta
        elif action == 1:
            return 0
        else:
            return self._v_delta

    def _reset(self) -> ts.StepType:
        logging.debug("Reset called")
        self._step_counter = 0
        self._episode_ended = False

        voltage = self._v0 or np.random.randint(
            int(self.pvarray.voc * 0.6), self.pvarray.voc
        )
        irradiance = self.weather_df["Irradiance"].iloc[0]
        temperature = self.weather_df["Temperature"].iloc[0]
        self._state = [voltage, irradiance, temperature]

        # logging.debug(f"Reset state: {self._state}")
        # logging.info(f"Step counter: {self._step_counter}")

        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action) -> ts.StepType:
        # logging.debug('Env Step')
        if self._episode_ended:
            return self._reset()

        sim_inputs = self._state[:]
        old_state = self._state[:]
        delta_v = self._get_delta_v(action)
        sim_inputs[0] += delta_v
        sim_inputs[0] = clip_num(sim_inputs[0], minimum=0, maximum=self.pvarray.voc)

        pv_sim_result = self.pvarray.simulate(*sim_inputs)
        voltage = pv_sim_result.voltage
        power = pv_sim_result.power

        self._step_counter += 1
        # weather_idx = np.random.randint(self.weather_df_len)
        irradiance = self.weather_df["Irradiance"].iloc[self._step_counter]
        temperature = self.weather_df["Temperature"].iloc[self._step_counter]

        # Save the new state
        self._state = [voltage, irradiance, temperature]

        if power <= 0:
            reward = REWARD_NEG_POWER
            if self._early_end:
                self._episode_ended = True
        else:
            reward = (power / self._reward_normalizer) ** 2

        # Last episode -> reward=0
        if self._step_counter >= self._max_episode_steps:
            self._episode_ended = True
            reward = 0

        logging.debug(
            f"Old state: {old_state}, New state: {self._state}, Power: {power}, Reward: {reward}, Step: {self._step_counter}, Action: {action},  dV: {delta_v}, Ended: {self._episode_ended}"
        )

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.float32), reward, discount=self._discount
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    pvarray = PVArray()
    weather_df = read_weather_csv(os.path.join("data", "toy_weather.csv"))

    env = PVEnvDiscV0(pvarray, weather_df, v_delta=0.1, discount=0.99)

    validate_py_environment(env, episodes=1)

    logging.info("\n\nDONE")
