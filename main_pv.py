import os

import torch

from src.pv_env import PVEnv, PVEnvDiscreteV1
from src.networks import DiscreteActorCriticNetwork
from src.agents import DiscreteActorCritic

PV_PARAMS_PATH = os.path.join("parameters", "pvarray_01.json")
WEATHER_PATH = os.path.join("data", "weather_sim_01.csv")

LEARNING_RATE = 0.001
ENTROPY_BETA = 0.001
GAMMA = 1
N_STEPS = 8
BATCH_SIZE = 16

if __name__ == "__main__":
    env = PVEnvDiscreteV1.from_file(
        PV_PARAMS_PATH,
        WEATHER_PATH,
        max_episode_steps=830,
        v_delta=0.2,
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
    # for _ in range(100):
    agent.train(steps=10000, verbose_every=100)
    # agent.plot_performance()
    # agent.exp_train_source.play_episode()
    # env.render()
    agent.exp_train_source.play_episode()
    env.render()