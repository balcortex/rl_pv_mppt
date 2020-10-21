import gym
import pandas as pd
from gym.envs.registration import EnvSpec
from gym.utils import seeding
from typing import Optional
import numpy as np
import collections
from dataclasses import dataclass, field
import os
import matplotlib.pyplot as plt

from src.pv_array import PVArray
from src.utils import load_dict, read_weather_csv

StepResult = collections.namedtuple(
    "StepResult", field_names=["obs", "reward", "done", "info"]
)


@dataclass
class History:
    irradiance: list = field(default_factory=list)
    cell_temp: list = field(default_factory=list)
    power: list = field(default_factory=list)
    voltage: list = field(default_factory=list)
    delta: list = field(default_factory=list)


class PVEnv(gym.Env):
    """
    PV Environment abstract class for solving the MPPT by reinforcement learning

    Parameters:
        - pvarray: the pvarray object
        - weather_df: a pandas dataframe object containing weather readings
        - discount: discount future rewards
            - 0: only account inmmediate reward
            - 1: all rewards weighs the same
        - max_episode_steps: maximum number of steps in the episode
            - 0: the episode last until the dataframe is exhausted
        - v0: initial load voltage
        - seed: for reproducibility
        - reset_on_neg: whether a negative power output finishes the episode
    """

    metadata = {"render.modes": ["human"]}
    spec = EnvSpec("PVEnv-v0")

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        discount: float,
        max_episode_steps: int,
        seed: Optional[int] = None,
        reset_on_neg: bool = True,
    ):
        assert isinstance(weather_df, pd.DataFrame)

        self.pvarray = pvarray
        self.weather = weather_df
        self.max_episode_steps = max_episode_steps
        self.discount = discount
        self.reset_on_neg = reset_on_neg

        if seed:
            np.random.seed(seed)

    def _reset(self):
        raise NotImplementedError

    def _step(self, action) -> np.ndarray:
        raise NotImplementedError

    def _get_delta_v(self, action: float) -> float:
        raise NotImplementedError

    @classmethod
    def from_file(cls, pv_params_path: str, weather_path: str, **kwargs):
        pvarray = PVArray.from_json(pv_params_path)
        weather = read_weather_csv(weather_path)
        return cls(pvarray, weather, **kwargs)

    # @property
    # def max_episode_steps(self) -> int:
    #     # return self._max_episode_steps or len(self.weather)
    #     return self._max_episode_steps


class PVEnvDiscrete(PVEnv):
    """
    PV discrete environment with availability of weather observations.

    Available actions:
        0: decrement v_delta
        1: do nothing
        2: increment v_delta

    Observations:
        [voltage, irradiance, cell temperature]

    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        v_delta: float,
        discount: float,
        max_episode_steps: int,
        seed: Optional[int] = None,
        reset_on_neg: bool = True,
    ) -> None:
        super().__init__(
            pvarray,
            weather_df,
            discount,
            max_episode_steps,
            seed,
            reset_on_neg,
        )

        self.v_delta = v_delta

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.pvarray.voc, 1200, 50]),
            shape=(3,),
            dtype=np.float32,
        )

    def _get_delta_v(self, action: float) -> float:
        if action == 0:
            return -self.v_delta
        elif action == 1:
            return 0.0
        elif action == 2:
            return self.v_delta
        else:
            raise ValueError(f"action must be [0, 1, 2], received={action}")

    def _add_history(self, p, v, g, t, dv) -> None:
        self.history.power.append(p)
        self.history.voltage.append(v)
        self.history.irradiance.append(g)
        self.history.cell_temp.append(t)
        self.history.delta.append(dv)

    def reset(self) -> np.ndarray:
        self.history = History()
        self.step_counter = 0
        self.step_idx = np.random.randint(0, len(self.weather) - self.max_episode_steps)
        self.done = False

        v = np.random.randint(int(self.pvarray.voc * 0.7), int(self.pvarray.voc * 0.9))
        g, t = self.weather[["Irradiance", "Temperature"]].iloc[self.step_idx]

        self.obs = [v, g, t]
        self._add_history(p=np.NaN, v=v, g=g, t=t, dv=np.NaN)
        return np.array(self.obs)

    def step(self, action: int) -> StepResult:
        if self.done:
            raise ValueError("The episode is done")

        self.step_idx += 1
        self.step_counter += 1

        delta_v = self._get_delta_v(action)
        v = np.clip(self.obs[0] + delta_v, 0, self.pvarray.voc)
        g, t = self.weather[["Irradiance", "Temperature"]].iloc[self.step_idx]
        self.obs = [v, g, t]
        pv_sim_result = self.pvarray.simulate(*self.obs)

        reward = pv_sim_result.power
        if pv_sim_result.power < 0:
            self.done = True
            reward = 0
        if self.step_counter >= self.max_episode_steps:
            self.done = True

        self._add_history(pv_sim_result.power, v, g, t, delta_v)

        return StepResult(
            self.obs,
            reward,
            self.done,
            {"step_idx": self.step_idx, "steps": self.step_counter},
        )

    def render(self) -> None:
        p_real, v_real, _ = self.pvarray.get_true_mpp(
            self.history.irradiance, self.history.cell_temp
        )
        plt.subplot(2, 3, 1)
        plt.plot(self.history.irradiance, label="Irradiance")
        plt.legend()
        plt.subplot(2, 3, 2)
        plt.plot(self.history.cell_temp, label="Cell temperature")
        plt.legend()
        plt.subplot(2, 3, 3)
        plt.plot(self.history.power, label="Power")
        plt.plot(p_real, label="Max")
        plt.legend()
        plt.subplot(2, 3, 4)
        plt.plot(self.history.voltage, label="Voltage")
        plt.plot(v_real, label="Vmpp")
        plt.legend()
        plt.subplot(2, 3, 5)
        plt.plot(self.history.delta, "o", label="Actions")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    pvenv = PVEnvDiscrete.from_file(
        pv_params_path=os.path.join("parameters", "pvarray_01.json"),
        weather_path=os.path.join("data", "weather_real_01.csv"),
        discount=0.99,
        max_episode_steps=5,
        v_delta=0.1,
    )

    obs = pvenv.reset()
    while True:
        action = pvenv.action_space.sample()
        new_obs, reward, done, info = pvenv.step(action)
        # print("obs", obs)
        # print("action", action)
        # print("new_obs", new_obs)
        # print("reward", reward)
        # print("done", done)
        # print("info", info)

        if done:
            break

    pvenv.render()