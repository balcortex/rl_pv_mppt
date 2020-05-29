import logging
import os

import numpy as np
import pandas as pd

from tf_agents.environments import py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step as ts

from typing import Optional

from pvmppt.data_manager import parse_depfie_csv
from pvmppt.pv_array import PVArray
from pvmppt.utils import clip_var


REWARD_NEG_POWER = -200


class PVEnv(py_environment.PyEnvironment):
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

        self._max_episode_steps = max_episode_steps or len(weather_df)
        self._discount = discount
        self._reward_normalizer = float(pvarray.model_params["Vm"]) * float(
            pvarray.model_params["Im"]
        )
        self._v0 = v0
        self._early_end = early_end

        if seed:
            np.random.seed(seed)

        self._action_spec = None
        self._observation_spec = None
        self._step_counter = 0

        logging.info(f"Max episodes: {self._max_episode_steps}")
        logging.info(f"Discount: {self._discount}")
        logging.info(f"Reward normalizer {self._reward_normalizer}")
        logging.info(f"Initial V: {self._v0}")
        logging.info(f"Early end: {self._early_end}")
        # logging.info(f"Action spec: {self.action_spec()}")
        # logging.info(f"Obs space: {self.observation_spec()}")

    def action_spec(self) -> BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> BoundedArraySpec:
        return self._observation_spec

    def _reset(self) -> ts.StepType:
        # v = v0 = self._v0 or np.random.randint(
        #     int(self.pvarray.voc * 0.8), self.pvarray.voc
        # )
        v = v0 = self._v0 or np.random.randint(0, self.pvarray.voc)
        logging.info(f"V0: {v0}")
        logging.debug(f"Voc: {self.pvarray.voc}")
        p = p0 = 0.0
        delta_v = 0.0
        irradiance = self.weather_df["Irradiance"].iloc[0]
        temperature = self.weather_df["Temperature"].iloc[0]
        # state = [v, p, delta_v, v_old, p_old, g, t]
        self._state = [v, p, delta_v, v0, p0, irradiance, temperature]
        logging.debug(f"Reset state: {self._state}")
        logging.info(f"Step counter: {self._step_counter}")
        self._step_counter = 0
        self._episode_ended = False

        return ts.restart(np.array(self._state, dtype=np.float32))

    def _get_delta_v(self, action: float) -> float:
        raise NotImplementedError

    def _step(self, action) -> ts.StepType:
        # logging.debug('Env Step')
        if self._episode_ended:
            return self._reset()

        delta_v = self._get_delta_v(action)
        logging.debug(f"Action: {action}, dV: {delta_v}")

        v_old = self._state[0]
        p_old = self._state[1]
        v = clip_var(v_old + delta_v, 0, self.pvarray.voc)

        # weather_idx = np.random.randint(self.weather_df_len)
        g = self.weather_df["Irradiance"].iloc[self._step_counter]
        t = self.weather_df["Temperature"].iloc[self._step_counter]

        pv_sim_result = self.pvarray.simulate(v, g, t)
        v = pv_sim_result.voltage
        p = pv_sim_result.power

        # state = [v, p, delta_v, v_old, p_old, g, t]
        self._state = [v, p, delta_v, v_old, p_old, g, t]
        reward = (p / self._reward_normalizer) ** 2
        self._step_counter += 1

        if p <= 0:
            if self._early_end:
                self._episode_ended = True
            self._state[1] = 0
            reward = REWARD_NEG_POWER

        if self._step_counter >= self._max_episode_steps:
            self._episode_ended = True

        # logging.debug(
        #     f"Idx: {self._step_counter:3}, G: {g:4}, T: {t:5.2f}, V: {v:5.2f}, P: {p:7.2f}, dV: {delta_v:4.1f}, Rew: {reward:12.6f}"
        # )

        logging.debug(f"Step state: {self._state}")

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.float32), reward, discount=self._discount
            )


class PVEnvDisc(PVEnv):
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
            pvarray, weather_df, discount, max_episode_steps, v0, seed, early_end,
        )

        self._v_delta = v_delta

        self._action_spec = BoundedArraySpec(
            shape=(), dtype=np.int32, name="action", minimum=0, maximum=2,
        )
        self._observation_spec = BoundedArraySpec(
            shape=(7,),
            dtype=np.float32,
            name="observation",
            minimum=[0, 0, -self._v_delta, 0, 0, 0, 0],
            maximum=[1e4, 1e4, self._v_delta, 1e4, 1e4, 1200, 50],
        )

        logging.info(f"Action spec: {self.action_spec()}")
        logging.info(f"Obs space: {self.observation_spec()}")
        logging.info(f"Delta V: {self._v_delta}")


class PVEnvDiscFullV0(PVEnvDisc):
    """
    PV discrete environment with early end, and availability of all observations.
    
    Available actions:
        0: decrement v_delta
        1: do nothing
        2: increment v_delta
    
    Observations:
        [v, p, delta_v, v_old, p_old, g, t]
        
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
    ) -> None:
        super().__init__(
            pvarray, weather_df, v_delta, discount, max_episode_steps, v0, seed, True,
        )

    def _get_delta_v(self, action: float) -> float:
        if action == 0:
            return -self._v_delta
        elif action == 1:
            return 0
        else:
            return self._v_delta


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    pv_params = {
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
    pvarray = PVArray(pv_params)
    weather_df_path = os.path.join("data", "toy_weather.csv")
    weather_df = parse_depfie_csv(weather_df_path)

    env = PVEnvDiscFullV0(pvarray, weather_df, v_delta=0.1, discount=0.99)

    validate_py_environment(env, episodes=1)

    logging.info("\n\nDONE")
