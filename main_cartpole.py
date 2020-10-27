import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.networks import ActorCriticNetwork
from src.agents import DiscreteActorCritic


GAMMA = 0.99
LEARNING_RATE = 0.01
ENTROPY_BETA = 0.001
N_STEPS = 8
BATCH_SIZE = 16


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    device = torch.device("cpu")
    net = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n).to(
        device
    )

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
    agent.train(800, verbose_every=100)
    agent.plot_performance()
