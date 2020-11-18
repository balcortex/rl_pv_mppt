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


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size: int, n_actions: int, hidden_units: int = 128):
        super().__init__()

        self.base1 = nn.Sequential(nn.Linear(input_size, hidden_units), nn.ReLU())
        self.base2 = nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
        self.base3 = nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
        self.mean = nn.Sequential(nn.Linear(hidden_units, n_actions), nn.Tanh())
        # self.mean = nn.Sequential(nn.Linear(hidden_units, n_actions))
        self.var = nn.Sequential(nn.Linear(hidden_units, n_actions), nn.Softplus())
        self.value = nn.Sequential(nn.Linear(hidden_units, 1))

    def forward(self, x):
        x = self.base1(x)
        x = self.base2(x)
        x = self.base3(x)
        return self.mean(x), self.var(x), self.value(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_units: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        return self.net(x)


class ActorNetwork(nn.Module):
    def __init__(self, input_size: int, n_actions: int, hidden_units=128):
        super().__init__()

        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
        )
        self.mean = nn.Sequential(
            nn.Linear(hidden_units, n_actions),
        )
        self.var = nn.Sequential(nn.Linear(hidden_units, n_actions), nn.Softplus())

    def forward(self, x):
        x = self.base(x)
        return self.mean(x), self.var(x), 1
