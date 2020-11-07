import gym
import pandas as pd
from gym.envs.registration import EnvSpec
from gym.utils import seeding
from typing import Optional, List
import numpy as np
import collections
from dataclasses import dataclass, field
import os
import matplotlib.pyplot as plt

from src.pv_array import PVArray
from src.utils import load_dict, read_weather_csv

G_MAX = 1200
T_MAX = 50
NEG_REWARD = -100

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


class PVEnvBase(gym.Env):
    """
    PV Environment abstract class for solving the MPPT by reinforcement learning

    Parameters:
        - pvarray: the pvarray object
        - weather_df: a pandas dataframe object containing weather readings
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
        seed: Optional[int] = None,
        reset_on_neg: bool = True,
        normalize: bool = False,
    ):
        assert isinstance(weather_df, pd.DataFrame)

        self.pvarray = pvarray
        self.weather = weather_df
        self.reset_on_neg = reset_on_neg
        self.normalize = normalize

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


class PVEnv(PVEnvBase):
    """
    PV environment with availability of all observations.
    Continuos actions

    Observations:
        [power, voltage, irradiance, cell temperature]

    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        seed: Optional[int] = None,
        reset_on_neg: bool = True,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            pvarray,
            weather_df,
            seed,
            reset_on_neg,
            normalize,
        )

        self.action_space = gym.spaces.Box(
            low=-10,
            high=10,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, 0, 0, 0]),
            high=np.array([self.pvarray.pmax, self.pvarray.voc, G_MAX, T_MAX]),
            shape=(4,),
            dtype=np.float32,
        )

    def _get_delta_v(self, action: float) -> float:
        return int(action)

    def _add_history(self, p, v, g, t, dv) -> None:
        self.history.power.append(p)
        self.history.voltage.append(v)
        self.history.irradiance.append(g)
        self.history.cell_temp.append(t)
        self.history.delta.append(dv)

    def reset(self) -> np.ndarray:
        self.history = History()
        self.step_counter = 0
        self.step_idx = 0
        self.done = False

        v = np.random.randint(int(self.pvarray.voc * 0.7), int(self.pvarray.voc * 0.9))
        g, t = self.weather[["Irradiance", "Temperature"]].iloc[self.step_idx]
        p, *_ = self.pvarray.simulate(v, g, t)

        self.obs = [p, v, g, t]
        self._add_history(p=p, v=v, g=g, t=t, dv=np.NaN)

        obs = np.array(self.obs)
        if self.normalize:
            obs /= np.array([self.pvarray.pmax, self.pvarray.voc, G_MAX, T_MAX])

        return obs

    def step(self, action: float) -> StepResult:
        if self.done:
            raise ValueError("The episode is done")

        self.step_idx += 1
        self.step_counter += 1

        delta_v = self._get_delta_v(action)
        v = np.clip(self.obs[1] + delta_v, 0, self.pvarray.voc)
        g, t = self.weather[["Irradiance", "Temperature"]].iloc[self.step_idx]
        p, *_ = self.pvarray.simulate(v, g, t)
        self.obs = [p, v, g, t]

        reward = p / self.pvarray.pmax
        if p < 0 or v < 1:
            self.done = True
            reward = NEG_REWARD
        if self.step_counter >= len(self.weather) - 1:
            self.done = True

        self._add_history(p, v, g, t, delta_v)

        obs = np.array(self.obs)
        if self.normalize:
            obs /= np.array([self.pvarray.pmax, self.pvarray.voc, G_MAX, T_MAX])

        return StepResult(
            obs,
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


class PVEnvDiscrete(PVEnv):
    """
    Discrete environment

    Observations:
    [power, voltage, irradiance, cell temperature]
    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        seed: Optional[int] = None,
        reset_on_neg: bool = True,
        normalize: bool = False,
        actions: List[float] = [-0.1, 0.0, 0.1],
    ) -> None:
        super().__init__(
            pvarray,
            weather_df,
            seed,
            reset_on_neg,
            normalize,
        )

        self.actions = actions
        self.action_space = gym.spaces.Discrete(len(actions))

    def _get_delta_v(self, action: int) -> float:
        return self.actions[action]


class PVEnvDiscreteDiffV1(PVEnvDiscrete):
    """
    Discrete environment

    Observations:
    [delta_p, delta_v, v, irradiance, cell temperature]
    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        seed: Optional[int] = None,
        reset_on_neg: bool = True,
        normalize: bool = False,
        actions: List[float] = [-0.1, 0.0, 0.1],
    ) -> None:
        super().__init__(
            pvarray,
            weather_df,
            seed,
            reset_on_neg,
            normalize,
            actions,
        )

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 5),
            high=np.array([np.inf] * 5),
            shape=(5,),
            dtype=np.float32,
        )

    def reset(self):
        self.obs_delta = super().reset()
        return np.array(
            [0.0, 0.0, self.obs_delta[1], self.obs_delta[2], self.obs_delta[3]]
        )

    def step(self, action: float):
        obs, reward, done, info = super().step(action)

        delta_p = obs[0] - self.obs_delta[0]
        delta_v = obs[1] - self.obs_delta[1]

        self.obs_delta = obs

        return np.array([delta_p, delta_v, obs[1], obs[2], obs[3]]), reward, done, info


class PVEnvDiscreteDiffV2(PVEnvDiscreteDiffV1):
    """
    Discrete environment

    Observations:
    [delta_p, delta_v]
    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        seed: Optional[int] = None,
        reset_on_neg: bool = True,
        normalize: bool = False,
        actions: List[float] = [-0.1, 0.0, 0.1],
    ) -> None:
        super().__init__(
            pvarray,
            weather_df,
            seed,
            reset_on_neg,
            normalize,
            actions,
        )

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 2),
            high=np.array([np.inf] * 2),
            shape=(2,),
            dtype=np.float32,
        )

    def reset(self):
        obs = super().reset()
        return np.array([obs[0], obs[1]])

    def step(self, action: float):
        obs, _, done, info = super().step(action)

        if obs[0] < -1:
            reward = -1
        elif obs[0] >= -1 and obs[0] < 1:
            reward = 0
        else:
            reward = 0.1

        return np.array([obs[0], obs[1]]), reward, done, info

    # def reset(self) -> np.ndarray:
    #     obs = super().reset() * (1 / np.array([self.pvarray.voc, G_MAX, T_MAX]))
    #     return np.array(obs)

    # def step(self, action: int) -> StepResult:
    #     result = super().step(action)
    #     obs = result.obs * (1 / np.array([self.pvarray.voc, G_MAX, T_MAX]))
    #     return StepResult(np.array(obs), result.reward, result.done, result.info)


if __name__ == "__main__":

    env = PVEnvDiscreteDiffV1.from_file(
        pv_params_path=os.path.join("parameters", "pvarray_01.json"),
        weather_path=os.path.join("data", "weather_sim_01.csv"),
        normalize=True,
        actions=[-0.1, 0, 0.1],
    )

    obs = env.reset()
    # while True:
    #     action = env.action_space.sample()
    #     new_obs, reward, done, info = env.step(action)

    #     if done:
    #         break

    # env.render()