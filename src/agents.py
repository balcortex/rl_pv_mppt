from typing import Optional, List

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from src.experience import ExperienceSorceDiscountedSteps
from src.policies import (
    BasePolicy,
    DiscreteCategoricalDistributionPolicy,
    DiscreteGreedyPolicy,
)
from src.logger import logger


class AgentABC:
    "Abstract class of Agent"

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def test(self, num_episodes: int):
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def plot_performance(self, metrics: List[str]) -> None:
        raise NotImplementedError

    def save(self, path: Optional[str]) -> None:
        raise NotImplementedError

    def load(self, path: Optional[str]) -> None:
        raise NotImplementedError

    def _get_train_policy(self) -> BasePolicy:
        raise NotImplementedError

    def _get_test_policy(self) -> BasePolicy:
        raise NotImplementedError


class Agent(AgentABC):
    "Base class for an Agent"

    def __init__(
        self,
        env: gym.Env,
        test_env: gym.Env,
        net: nn.Module,
        device: torch.device,
        gamma: float,
        beta_entropy: float,
        lr: float,
        n_steps: int,
        batch_size: int,
        chk_path: str,
        optimizer: str = "adam",
    ):
        self.env = env
        self.test_env = test_env
        self.net = net
        self.device = device
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.n_steps = n_steps
        self.chk_path = chk_path
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        else:
            raise ValueError("Only `adam` is supported")
        self.policy = self._get_train_policy()
        self.test_policy = self._get_test_policy()
        self.exp_train_source = ExperienceSorceDiscountedSteps(
            env=env,
            policy=self.policy,
            gamma=gamma,
            n_steps=n_steps,
            steps=batch_size,
        )
        self.exp_test_source = ExperienceSorceDiscountedSteps(
            env=test_env,
            policy=self.test_policy,
            gamma=gamma,
            n_steps=n_steps,
            steps=batch_size,
        )

        self.reset()

        if chk_path:
            if os.path.exists(chk_path):
                self.load()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.policy(obs)

    def test(self, num_episodes: int = 1):
        episodes = self.exp_test_source.play_episodes(episodes=num_episodes)

        reward = 0
        for ep in episodes:
            for step in ep:
                reward += step.reward
        reward /= num_episodes

        return reward

    def reset(self) -> None:
        self.hist_total_rew = []
        self.hist_mean_rew = []
        self.hist_steps = []
        self.hist_total_loss = []
        self.hist_entropy_loss = []
        self.hist_value_loss = []
        self.hist_policy_loss = []
        self.counter_ep = 0
        self.counter_steps_per_ep = 0
        self.counter_step = 0
        self.ep_reward = 0
        self.total_loss = np.NaN
        self.policy_loss = np.NaN
        self.entropy_loss = np.NaN
        self.value_loss = np.NaN
        self.mean_reward = np.NaN
        self.steps_per_ep = np.NaN

    def plot_performance(
        self,
        metrics: List[str] = [
            "mean_rewards",
            "total_rewards",
            "total_loss",
            "policy_loss",
            "value_loss",
            "entropy_loss",
        ],
    ) -> None:
        dic = {
            "mean_rewards": self.hist_mean_rew,
            "total_rewards": self.hist_total_rew,
            "total_loss": self.hist_total_loss,
            "policy_loss": self.hist_policy_loss,
            "value_loss": self.hist_value_loss,
            "entropy_loss": self.hist_entropy_loss,
        }

        for metric in metrics:
            plt.plot(self.hist_steps, dic[metric], label=metric)
            plt.legend()
            plt.show()

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.chk_path
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hist_total_rew": self.hist_total_rew,
                "hist_mean_rew": self.hist_mean_rew,
                "hist_steps": self.hist_steps,
                "hist_total_loss": self.hist_total_loss,
                "hist_entropy_loss": self.hist_entropy_loss,
                "hist_value_loss": self.hist_value_loss,
                "hist_policy_loss": self.hist_policy_loss,
                "counter_ep": self.counter_ep,
                "counter_steps_per_ep": self.counter_steps_per_ep,
                "counter_step": self.counter_step,
                "total_loss": self.total_loss,
                "policy_loss": self.policy_loss,
                "entropy_loss": self.entropy_loss,
                "value_loss": self.value_loss,
                "mean_reward": self.mean_reward,
                "steps_per_ep": self.steps_per_ep,
            },
            path,
        )
        logger.info(f"Checkpoint saved to {path}")

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.chk_path
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.hist_total_rew = checkpoint["hist_total_rew"]
        self.hist_mean_rew = checkpoint["hist_mean_rew"]
        self.hist_steps = checkpoint["hist_steps"]
        self.hist_total_loss = checkpoint["hist_total_loss"]
        self.hist_entropy_loss = checkpoint["hist_entropy_loss"]
        self.hist_value_loss = checkpoint["hist_value_loss"]
        self.hist_policy_loss = checkpoint["hist_policy_loss"]
        self.counter_ep = checkpoint["counter_ep"]
        self.counter_steps_per_ep = checkpoint["counter_steps_per_ep"]
        self.counter_step = checkpoint["counter_step"]
        self.ep_reward = 0
        self.total_loss = checkpoint["total_loss"]
        self.policy_loss = checkpoint["policy_loss"]
        self.entropy_loss = checkpoint["entropy_loss"]
        self.value_loss = checkpoint["value_loss"]
        self.mean_reward = checkpoint["mean_reward"]
        self.steps_per_ep = checkpoint["steps_per_ep"]
        logger.info(f"Checkpoint loaded from {path}")


class DiscreteActorCritic(Agent):
    """
    Agent that has a network that predicts both the action probabilities and the value
    of the state. The value is used to calculate the Advantage (A) of and action given
    the state -> A(s,a) = Q(s,a) - V(s).

    Parameters:


    """

    def __init__(
        self,
        env: gym.Env,
        test_env: gym.Env,
        net: nn.Module,
        device: torch.device,
        gamma: float,
        beta_entropy: float,
        lr: float,
        n_steps: int,
        batch_size: int,
        chk_path: str,
        optimizer: str = "adam",
        apply_softmax: bool = True,
    ):
        self.apply_softmax = apply_softmax
        super().__init__(
            env=env,
            test_env=test_env,
            net=net,
            device=device,
            gamma=gamma,
            beta_entropy=beta_entropy,
            lr=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            chk_path=chk_path,
            optimizer=optimizer,
        )

    def train(
        self,
        steps: int,
        verbose_every: Optional[int] = 0,
        save_every: Optional[int] = 0,
    ):

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
                    self.steps_per_ep = self.counter_steps_per_ep

                    self.hist_total_rew.append(self.ep_reward)
                    self.mean_reward = np.mean(self.hist_total_rew[-100:])
                    self.hist_mean_rew.append(self.mean_reward)
                    self.hist_steps.append(self.counter_step)
                    self.hist_total_loss.append(self.total_loss)
                    self.hist_entropy_loss.append(self.entropy_loss)
                    self.hist_value_loss.append(self.value_loss)
                    self.hist_policy_loss.append(self.policy_loss)

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
            # std, mean = torch.std_mean(values_target_t)
            # values_target_t -= mean
            # values_target_t /= std + 1e-6

            self.optimizer.zero_grad()
            logits_t, values_t = self.net(states_t)
            values_t = values_t.squeeze()
            loss_value_t = F.mse_loss(values_target_t, values_t)
            self.value_loss = loss_value_t.item()

            log_prob_actions_t = F.log_softmax(logits_t, dim=1)
            log_prob_chosen = log_prob_actions_t.gather(
                1, actions_t.unsqueeze(1)
            ).squeeze()
            advantage_t = values_target_t - values_t.detach()
            loss_policy_t = (log_prob_chosen * advantage_t).mean()
            self.policy_loss = loss_policy_t.item()

            prob_actions_t = F.softmax(logits_t, dim=1)
            entropy_t = -(prob_actions_t * log_prob_actions_t).sum(dim=1).mean()
            loss_entropy_t = -self.beta_entropy * entropy_t
            self.entropy_loss = loss_entropy_t.item()

            loss_total_t = -loss_policy_t + loss_value_t + loss_entropy_t
            loss_total_t.backward()
            self.optimizer.step()
            self.total_loss = loss_total_t.item()

            if verbose_every:
                if self.counter_step % verbose_every == 0:
                    print(
                        "\n",
                        f"{self.counter_step}: loss={self.total_loss:.6f}, ",
                        f"mean reward={self.mean_reward:.2f}, ",
                        f"steps/ep={self.steps_per_ep}, ",
                        f"episodes={self.counter_ep}",
                    )

            if save_every and self.chk_path:
                if self.counter_step % save_every == 0:
                    self.save()

        if self.chk_path:
            self.save()

    def _get_train_policy(self) -> BasePolicy:
        return DiscreteCategoricalDistributionPolicy(
            net=self.net,
            device=self.device,
            apply_softmax=self.apply_softmax,
            net_index=0,
            add_batch_dim=True,
        )

    def _get_test_policy(self) -> BasePolicy:
        return DiscreteGreedyPolicy(
            net=self.net,
            device=self.device,
            net_index=0,
            add_batch_dim=True,
        )


class ContinuosActorCritic(DiscreteActorCritic):
    """
    Agent that has a network that predicts both the action probabilities and the value
    of the state. The value is used to calculate the Advantage (A) of and action given
    the state -> A(s,a) = Q(s,a) - V(s).
    """

    def __init__(
        self,
        env: gym.Env,
        test_env: gym.Env,
        net: nn.Module,
        device: torch.device,
        gamma: float,
        beta_entropy: float,
        lr: float,
        n_steps: int,
        batch_size: int,
        chk_path: str,
        optimizer: str = "adam",
    ):
        super().__init__(
            env=env,
            test_env=test_env,
            net=net,
            device=device,
            gamma=gamma,
            beta_entropy=beta_entropy,
            lr=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            chk_path=chk_path,
            optimizer=optimizer,
        )

    # def train

    def _get_train_policy(self, apply_softmax: bool) -> BasePolicy:
        return DiscreteCategoricalDistributionPolicy(
            net=self.net,
            device=self.device,
            apply_softmax=apply_softmax,
            net_index=0,
            add_batch_dim=True,
        )

    def _get_test_policy(self) -> BasePolicy:
        return DiscreteGreedyPolicy(
            net=self.net,
            device=self.device,
            net_index=0,
            add_batch_dim=True,
        )
