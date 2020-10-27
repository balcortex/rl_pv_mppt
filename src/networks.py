import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteActorCriticNetwork(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.net(x)
        return self.actor(x), self.critic(x)
