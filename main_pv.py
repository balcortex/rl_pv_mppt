import os

import torch

from src.pv_env import History, PVEnvDiscrete
from src.networks import DiscreteActorCriticNetwork
from src.agents import DiscreteActorCritic

PV_PARAMS_PATH = os.path.join("parameters", "pvarray_01.json")
WEATHER_PATH = os.path.join("data", "weather_sim_01.csv")

LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.000
GAMMA = 0.8
N_STEPS = 4
BATCH_SIZE = 16

if __name__ == "__main__":

    def reward_fn1(history: History) -> float:
        dp = history.dp[-1]

        if dp < -0.1:
            return -1
        elif -0.1 <= dp < 0.1:
            return 0
        else:
            return 1

    def reward_fn2(history: History) -> float:
        dp = history.dp[-1]

        if dp < 0:
            return 1 * dp
        else:
            return 2 * dp

    env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,
        WEATHER_PATH,
        states=["dv", "dp"],
        reward_fn=reward_fn2,
        actions=[-1.0, -0.1, 0, 0.1, 1.0],
    )
    device = torch.device("cpu")
    net = DiscreteActorCriticNetwork(
        input_size=env.observation_space.shape[0], n_actions=env.action_space.n
    ).to(device)
    agent = DiscreteActorCritic(
        env=env,
        net=net,
        device=device,
        gamma=GAMMA,
        beta_entropy=ENTROPY_BETA,
        lr=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
    )
    # for _ in range(10):
    agent.train(steps=10000, verbose_every=100)
    # agent.plot_performance()
    # agent.exp_train_source.play_episode()
    # env.render()
    agent.exp_train_source.play_episode()
    env.render()
    # agent.plot_performance()
