from typing import Optional, List

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.experience import ExperienceSorceDiscountedSteps
from src.policies import BasePolicy, DiscreteCategoricalDistributionPolicy


class BasePolicy:
    "Abstract class of Agent"

    def __call__(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def _reset(self):
        self.hist_total_rew = []
        self.hist_mean_rew = []
        self.hist_steps = []
        self.hist_total_loss = []
        self.counter_ep = 0
        self.counter_steps_per_ep = 0
        self.counter_step = 0
        self.ep_reward = 0
        self.loss = np.NaN


class DiscreteActorCritic(BasePolicy):
    """
    Agent that has a network that predicts both the action probabilities and the value
    of the state. The value is used to calculate the Advantage (A) of and action given
    the state -> A(s,a) = Q(s,a) - V(s).

    Parameters:


    """

    def __init__(
        self,
        env: gym.Env,
        net: nn.Module,
        device: torch.device,
        gamma: float,
        beta_entropy: float,
        lr: float,
        n_steps: int,
        batch_size: int,
        optimizer: str = "adam",
        apply_softmax: bool = True,
    ):
        self.env = env
        self.net = net
        self.device = device
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.n_steps = n_steps
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        else:
            raise ValueError("Only `adam` is supported")
        self.policy = DiscreteCategoricalDistributionPolicy(
            net=net, device=device, apply_softmax=apply_softmax, net_index=0
        )
        self.exp_train_source = ExperienceSorceDiscountedSteps(
            env=env, policy=self.policy, gamma=gamma, n_steps=n_steps, steps=batch_size
        )

        self._reset()

    def train(self, steps: int, verbose_every: Optional[int] = 0):

        mean_reward = np.NaN
        steps_per_ep = np.NaN

        for _ in tqdm(range(steps)):
            self.counter_step += 1

            states = []
            actions = []
            last_states = []
            values = []
            dones = []

            for exp in next(self.exp_train_source):
                self.ep_reward += exp.reward
                self.counter_steps_per_ep += exp.steps

                states.append(exp.state)
                actions.append(exp.action)
                values.append(exp.discounted_reward)

                if exp.last_state is None:
                    dones.append(True)
                    last_states.append(
                        exp.state
                    )  # as a placeholder, we'll mask this val

                    self.counter_ep += 1
                    steps_per_ep = self.counter_steps_per_ep

                    self.hist_total_rew.append(self.ep_reward)
                    mean_reward = np.mean(self.hist_total_rew[-100:])
                    self.hist_mean_rew.append(mean_reward)
                    self.hist_steps.append(self.counter_step)
                    self.hist_total_loss.append(self.loss)

                    self.ep_reward = 0
                    self.counter_steps_per_ep = 0
                else:
                    dones.append(False)
                    last_states.append(exp.last_state)

            last_states_t = torch.tensor(last_states, dtype=torch.float32).to(
                self.device
            )
            states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions_t = torch.tensor(actions, dtype=torch.int64).to(self.device)

            with torch.no_grad():
                values_last = (
                    self.net(last_states_t)[1].squeeze() * self.gamma ** self.n_steps
                )
            values_last[dones] = 0  # the value of terminal states is zero

            # Normalize the rewards
            values_target_t = torch.tensor(values, dtype=torch.float32).to(self.device)
            std, mean = torch.std_mean(values_target_t)
            values_target_t -= mean
            values_target_t /= std + 1e-6

            self.optimizer.zero_grad()
            logits_t, values_t = self.net(states_t)
            values_t = values_t.squeeze()
            loss_value_t = F.mse_loss(values_target_t, values_t)

            log_prob_actions_t = F.log_softmax(logits_t, dim=1)
            log_prob_chosen = log_prob_actions_t.gather(
                1, actions_t.unsqueeze(1)
            ).squeeze()
            advantage_t = values_target_t - values_t.detach()
            loss_policy_t = (log_prob_chosen * advantage_t).mean()

            prob_actions_t = F.softmax(logits_t, dim=1)
            entropy_t = (prob_actions_t * log_prob_actions_t).sum(dim=1).mean()
            loss_entropy_t = self.beta_entropy * entropy_t

            loss_total_t = -loss_policy_t + loss_value_t + loss_entropy_t
            loss_total_t.backward()
            self.optimizer.step()
            self.loss = loss_total_t.item()

            if verbose_every:
                if self.counter_step % verbose_every == 0:
                    print(
                        f"{self.counter_step}: loss={self.loss:.6f}, ",
                        f"mean reward={mean_reward:.2f}, ",
                        f"steps/ep={steps_per_ep}, ",
                        f"episodes={self.counter_ep}",
                    )

    def plot_performance(self, metrics: List[str]) -> None:
        if "loss" in metrics:
            plt.plot(self.hist_steps, self.hist_total_loss, label="Total loss")
            plt.legend()
            plt.show()

        if "mean_rewards" in metrics:
            plt.plot(self.hist_steps, self.hist_mean_rew, label="Mean rewards")
            plt.legend()
            plt.show()

        if "total_rewards" in metrics:
            plt.plot(self.hist_steps, self.hist_total_rew, label="Total rewards")
            plt.legend()
            plt.show()
