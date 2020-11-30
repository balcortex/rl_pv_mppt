from typing import Optional, Sequence, Union, Tuple
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import copy

from src.ou_noise import OUNoise
from src.policies import BasePolicy
from src.agents import Agent
from src.replay_buffer import ReplayBuffer


class DDPGActor(nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        hidden: Sequence[int] = (128,),
        use_tanh: bool = True,
    ):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(obs_size, hidden[0]),
            nn.ReLU(),
        )
        self.hidden = nn.ModuleList()
        if len(hidden) > 1:
            self.hidden.extend(
                nn.Sequential(
                    nn.Linear(inp, outp),
                    nn.ReLU(),
                )
                for (inp, outp) in zip(hidden, hidden[1:])
            )

        self.output = (
            nn.Sequential(nn.Linear(hidden[-1], act_size))
            if not use_tanh
            else nn.Sequential(nn.Linear(hidden[-1], act_size), nn.Tanh())
        )

    def forward(self, obs):
        obs = self.input(obs)
        for hid in self.hidden:
            obs = hid(obs)
        return self.output(obs)


class DDPGCritic(nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        hidden1: int = 128,
        hidden2: int = 128,
    ):
        super().__init__()

        self.obs_input = nn.Sequential(
            nn.Linear(obs_size, hidden1),
            nn.ReLU(),
        )
        self.hidden = nn.Sequential(
            nn.Linear(hidden1 + act_size, hidden2),
            nn.ReLU(),
        )

        self.output = nn.Sequential(nn.Linear(hidden2, 1))

    def forward(self, obs, action):
        x = self.obs_input(obs)
        x = self.hidden(torch.cat([x, action], dim=1))
        return self.output(x)


class DDPGPolicy(BasePolicy):
    def __init__(
        self,
        net: nn.Module,
        env: gym.Env,
        noise: OUNoise,
        device: Union[str, torch.device],
        add_batch_dim: bool = False,
        epsilon: float = 1.0,
    ):
        self.net = net
        self.device = device
        self.env = env
        self.noise = noise
        self.add_batch_dim = add_batch_dim
        self.epsilon = epsilon
        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]

    @torch.no_grad()
    def __call__(self, states: np.ndarray):
        noise = 0.0
        if self.epsilon > np.random.rand():
            # noise = self.noise.sample() * self.epsilon
            noise = self.noise.sample()
            # print(f"{noise=}")

        if self.add_batch_dim:
            states = states[np.newaxis, :]

        states_v = torch.tensor(states, dtype=torch.float32)
        actions = self.net(states_v).cpu().numpy()
        # print(f"{actions=}")
        actions += noise
        # print(f"{actions=}")
        actions = self._unscale_actions(actions)
        actions = actions.clip(
            self.env.action_space.low[0], self.env.action_space.high[0]
        )
        # print(f"{actions=}")

        if self.add_batch_dim:
            return actions[0]
        return actions

    def _unscale_actions(self, scled_actions: torch.Tensor) -> torch.Tensor:
        return self.low + (scled_actions + 1) * (self.high - self.low) / (2)


class TargetNet:
    "Wrapper around model which provides copy of it instead of trained weights"

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def __call__(self, x: torch.Tensor):
        return self.target_model(x)

    def sync(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha: float) -> None:
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class EpsilonDecayLinear:
    """
    Decay epsilon linearly

    Pameters:
        eps_start: epsilon initial value
        eps_final: epsilon final value
        max_steps: decrement epsilon during the number of steps
        current_step: state of the current step

    """

    def __init__(
        self,
        eps_start: float = 1.0,
        eps_final: float = 0.01,
        max_steps: int = 10_000,
        current_step: int = 0,
    ):
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.max_steps = max_steps
        self._current_step = current_step

    def __call__(self, step):
        return max(self.eps_final, self.eps_start - step / self.max_steps)

    def step(self, steps: int = 1) -> float:
        "Increment the step counter and return the current value of epsilon"
        self._current_step += steps
        return self.value

    def set_current_step(self, step: int) -> float:
        "Set the current step and return the current value"
        self._current_step = step
        return self.value

    @property
    def value(self) -> float:
        "Return the current value of epsilon"
        return self(self.current_step)

    @property
    def current_step(self) -> int:
        "Return the current step"
        return self._current_step


class DDPGAgent(Agent):
    def __init__(
        self,
        env: gym.Env,
        test_env: gym.Env,
        act_net: nn.Module,
        crt_net: nn.Module,
        noise: OUNoise,
        eps_schedule: EpsilonDecayLinear,
        device: torch.device,
        gamma: float,
        lr: float,
        n_steps: int,
        batch_size: int,
        chk_path: str,
        optimizer: str = "adam",
        norm_rewards: bool = False,
        clip_grads: Optional[float] = 0.0,
        crt_lr: float = 1e-2,
        tau: float = 1e-3,
    ):
        self.crt_net = crt_net
        self.crt_optimizer = torch.optim.Adam(self.crt_net.parameters(), lr=crt_lr)
        self.tgt_act = TargetNet(act_net)
        self.tgt_crt = TargetNet(crt_net)
        self.replay_buffer = ReplayBuffer(50_000)
        self.eps_schedule = eps_schedule
        self.batch_size = batch_size
        self.noise = noise
        self.norm_rewards = norm_rewards
        self.clip_grads = clip_grads
        self.tau = tau

        super().__init__(
            env=env,
            test_env=test_env,
            net=act_net,
            device=device,
            gamma=gamma,
            beta_entropy=None,
            lr=lr,
            n_steps=n_steps,
            batch_size=1,
            chk_path=chk_path,
            optimizer=optimizer,
        )
        self.optimizer = torch.optim.Adam(
            act_net.parameters(),
            lr=lr,
            # weight_decay=0.1,
            # weight_decay=0.1,
        )

    def _prepare_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for exp in next(self.exp_train_source):
            self.ep_reward += exp.reward
            self.counter_steps_per_ep += exp.steps

            if exp.last_state is None:
                done = True
                last_state = exp.state

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
                done = False
                last_state = exp.last_state

            self.replay_buffer.append(
                (exp.state, exp.action, exp.discounted_reward, done, last_state)
            )

        if len(self.replay_buffer) < self.batch_size:
            self._prepare_batch()

        self.eps_schedule.step()
        self.policy.epsilon = self.eps_schedule.value
        states, actions, values, dones, last_states = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(self.device)
        values_t = torch.tensor(values, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool).to(self.device)
        last_states_t = torch.tensor(last_states, dtype=torch.float32).to(self.device)

        if self.norm_rewards:
            std, mean = torch.std_mean(values_t)
            values_t -= mean
            values_t /= std + 1e-6

        return states_t, actions_t, values_t, dones_t, last_states_t

    def train_net(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_states: torch.Tensor,
    ) -> None:

        self.crt_optimizer.zero_grad()
        q = self.crt_net(states, actions)
        last_actions = self.tgt_act.target_model(last_states)
        q_last = self.tgt_crt.target_model(last_states, last_actions)
        q_last[dones] = 0.0

        q_ref = values.unsqueeze(-1) + q_last * self.gamma ** self.n_steps
        loss_value = F.mse_loss(q, q_ref.detach())
        self.value_loss = loss_value.item()
        loss_value.backward()
        # if self.clip_grads > 0:
        #     torch.nn.utils.clip_grad_norm_(self.crt_net.parameters(), self.clip_grads)
        self.crt_optimizer.step()

        self.crt_optimizer.zero_grad()
        self.optimizer.zero_grad()
        pred_actions = self.net(states)
        loss_policy = -self.crt_net(states, pred_actions).mean()
        self.policy_loss = loss_policy.item()
        loss_policy.backward()

        if self.clip_grads > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grads)
        self.optimizer.step()

        self.total_loss = self.policy_loss + self.value_loss

        self.tgt_act.alpha_sync(alpha=self.tau)
        self.tgt_crt.alpha_sync(alpha=self.tau)

    def _get_train_policy(self) -> BasePolicy:
        return DDPGPolicy(
            net=self.net,
            env=self.env,
            noise=self.noise,
            device=self.device,
            epsilon=1.0,
        )

    def _get_test_policy(self) -> BasePolicy:
        return DDPGPolicy(
            net=self.net,
            env=self.test_env,
            noise=None,
            device=self.device,
            epsilon=0.0,
        )


if __name__ == "__main__":
    device = torch.device("cpu")
    env = gym.make("Pendulum-v0")
    test_env = gym.make("Pendulum-v0")
    act_net = DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    crt_net = DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0])
    agent = DDPGAgent(
        env,
        test_env,
        act_net,
        crt_net,
        device,
        gamma=0.99,
        beta_entropy=0.001,
        lr=0.001,
        n_steps=4,
        batch_size=16,
        chk_path=None,
    )
    agent._prepare_batch()
