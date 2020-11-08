import gym
import pandas as pd
from typing import Optional, List
import numpy as np
import collections
from dataclasses import dataclass, field
import os
import matplotlib.pyplot as plt

from src.pv_array import PVArray
from src.utils import read_weather_csv

G_MAX = 1200
T_MAX = 60
NEG_REWARD = -100

StepResult = collections.namedtuple(
    "StepResult", field_names=["obs", "reward", "done", "info"]
)


@dataclass
class History:
    g: list = field(default_factory=list)
    t: list = field(default_factory=list)
    p: list = field(default_factory=list)
    v: list = field(default_factory=list)
    i: list = field(default_factory=list)
    dp: list = field(default_factory=list)
    dv: list = field(default_factory=list)
    di: list = field(default_factory=list)
    g_norm: list = field(default_factory=list)
    t_norm: list = field(default_factory=list)
    p_norm: list = field(default_factory=list)
    v_norm: list = field(default_factory=list)
    i_norm: list = field(default_factory=list)


class PVEnvBase(gym.Env):
    "PV Environment abstract class for solving the MPPT by reinforcement learning"
    # metadata = {"render.modes": ["human"]}
    # spec = gym.envs.registration.EnvSpec("PVEnv-v0")

    def reset(self):
        raise NotImplementedError

    def step(self, action) -> np.ndarray:
        raise NotImplementedError

    def _get_observation_space(self) -> gym.Space:
        raise NotImplementedError

    def _get_action_space(self) -> gym.Space:
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
    PV Continuos Environment for solving the MPPT by reinforcement learning

    Parameters:
        - pvarray: the pvarray object
        - weather_df: a pandas dataframe object containing weather readings
        - states: list of states to return as observations
        - reward_fn: function that calculates the reward
        - seed: for reproducibility
    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        states: List[str],
        reward_fn: callable,
        seed: Optional[int] = None,
    ) -> None:

        self.pvarray = pvarray
        self.weather = weather_df
        self.states = states
        self.reward_fn = reward_fn
        if seed:
            np.random.seed(seed)

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    def reset(self) -> np.ndarray:
        self.history = History()
        self.step_counter = 0
        self.step_idx = 0
        self.done = False

        v = np.random.randint(int(self.pvarray.voc * 0.7), int(self.pvarray.voc * 0.9))

        return self._store_step(v)

    def step(self, action: float) -> StepResult:
        if self.done:
            raise ValueError("The episode is done")

        self.step_idx += 1
        self.step_counter += 1

        delta_v = self._get_delta_v(action)
        v = np.clip(self.v + delta_v, 0, self.pvarray.voc)
        obs = self._store_step(v)
        reward = self.reward_fn(self.history)

        if self.history.p[-1] < 0 or self.history.v[-1] < 1:
            self.done = True
        if self.step_counter >= len(self.weather) - 1:
            self.done = True

        return StepResult(
            obs,
            reward,
            self.done,
            {"step_idx": self.step_idx, "steps": self.step_counter},
        )

    def render(self) -> None:
        p_real, v_real, _ = self.pvarray.get_true_mpp(self.history.g, self.history.t)
        plt.subplot(2, 3, 1)
        plt.plot(self.history.g, label="Irradiance")
        plt.legend()
        plt.subplot(2, 3, 2)
        plt.plot(self.history.t, label="Cell temperature")
        plt.legend()
        plt.subplot(2, 3, 3)
        plt.plot(self.history.p, label="Power")
        plt.plot(p_real, label="Max")
        plt.legend()
        plt.subplot(2, 3, 4)
        plt.plot(self.history.v, label="Voltage")
        plt.plot(v_real, label="Vmpp")
        plt.legend()
        plt.subplot(2, 3, 5)
        plt.plot(self.history.dv, "o", label="Delta V")
        plt.legend()
        plt.subplot(2, 3, 6)
        plt.plot(self.history.dp, "o", label="Delta P")
        plt.legend()
        plt.show()

    def _add_history(self, p, v, i, g, t) -> None:
        self.history.p.append(p)
        self.history.v.append(v)
        self.history.i.append(i)
        self.history.g.append(g)
        self.history.t.append(t)
        self.history.p_norm.append(p / self.pvarray.pmax)
        self.history.v_norm.append(v / self.pvarray.voc)
        self.history.i_norm.append(i / self.pvarray.isc)
        self.history.g_norm.append(g / G_MAX)
        self.history.t_norm.append(t / T_MAX)

        if len(self.history.p) < 2:
            self.history.dp.append(0.0)
            self.history.dv.append(0.0)
            self.history.di.append(0.0)
        else:
            self.history.dp.append(self.history.p[-1] - self.history.p[-2])
            self.history.dv.append(self.history.v[-1] - self.history.v[-2])
            self.history.di.append(self.history.i[-1] - self.history.i[-2])

    def _get_delta_v(self, action: float) -> float:
        if isinstance(action, list):
            action = action[0]
        return action

    def _get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=np.array([-np.inf] * len(self.states)),
            high=np.array([-np.inf] * len(self.states)),
            shape=(len(self.states),),
            dtype=np.float32,
        )

    def _get_action_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )

    def _store_step(self, v: float) -> np.ndarray:
        g, t = self.weather[["Irradiance", "Temperature"]].iloc[self.step_idx]
        p, self.v, i = self.pvarray.simulate(v, g, t)
        self._add_history(p=p, v=self.v, i=i, g=g, t=t)

        # getattr(handler.request, 'GET') is the same as handler.request.GET
        return np.array([getattr(self.history, state)[-1] for state in self.states])


class PVEnvDiscrete(PVEnv):
    """
    PV Discrete Environment for solving the MPPT by reinforcement learning

    Parameters:
        - pvarray: the pvarray object
        - weather_df: a pandas dataframe object containing weather readings
        - states: list of states to return as observations
        - reward_fn: function that calculates the reward
        - seed: for reproducibility
    """

    def __init__(
        self,
        pvarray: PVArray,
        weather_df: pd.DataFrame,
        states: List[str],
        reward_fn: callable,
        actions: List[float],
        seed: Optional[int] = None,
    ) -> None:
        self.actions = actions
        super().__init__(
            pvarray,
            weather_df,
            states,
            reward_fn,
            seed,
        )

    def _get_delta_v(self, action: int) -> float:
        return self.actions[action]

    def _get_action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.actions))


if __name__ == "__main__":

    def reward_fn(history: History) -> float:
        dp = history.dp[-1]
        if dp < -0.1:
            return -1
        elif -0.1 <= dp < 0.1:
            return 0
        else:
            return 1

    env = PVEnvDiscrete.from_file(
        pv_params_path=os.path.join("parameters", "pvarray_01.json"),
        weather_path=os.path.join("data", "weather_sim_01.csv"),
        states=["v", "p", "g", "t"],
        reward_fn=reward_fn,
        actions=[-0.1, 0, 0.1],
    )

    obs = env.reset()
    # while True:
    #     action = env.action_space.sample()
    #     new_obs, reward, done, info = env.step(action)

    #     if done:
    #         break

    # env.render()