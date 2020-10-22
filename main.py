import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import time

from src.agents import ActorCritic
from src.networks import ActorCriticNetwork
from src.pv_array import PVArray
from src.pv_env import PVEnv, PVEnvDiscrete
from src.utils import read_weather_csv
from src.experience import ExperienceSorceDiscountedSteps

PV_PARAMS_PATH = os.path.join("parameters", "pvarray_01.json")
WEATHER_PATH = os.path.join("data", "weather_sim_01.csv")

LEARNING_RATE = 1e-4
ENTROPY_BETA = 1e-3
GAMMA = 0.99
N_STEPS = 4
BATCH_SIZE = 128

if __name__ == "__main__":
    env = PVEnvDiscrete.from_file(
        PV_PARAMS_PATH,
        WEATHER_PATH,
        max_episode_steps=830,
        v_delta=0.2,
    )
    device = torch.device("cpu")
    net = ActorCriticNetwork(
        input_size=env.observation_space.shape[0], n_actions=env.action_space.n
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    agent = ActorCritic(net, device=device, apply_softmax=True)
    exp_source = ExperienceSorceDiscountedSteps(
        env, agent, gamma=GAMMA, n_steps=N_STEPS, steps=BATCH_SIZE
    )

    # sys.exit()

    total_rewards = []
    mean_rewards = []
    step_indexes = []
    losses = []
    loss = np.NaN
    ep_reward = 0.0
    ep_counter = 0
    steps_per_episode_counter = 0
    steps_per_episode = 0
    step_counter = 0
    mean_reward = np.NaN

    for _ in tqdm(list(range(10000))):
        step_counter += 1

        batch_states = []
        batch_actions = []
        batch_last_states = []
        batch_values = []
        batch_dones = []

        for exp in next(exp_source):
            ep_reward += exp.reward
            steps_per_episode_counter += exp.steps

            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            batch_values.append(exp.discounted_reward)

            if exp.last_state is None:
                batch_dones.append(True)
                batch_last_states.append(exp.state)  # we'll going to mask these anyway

                # env.render()
                ep_counter += 1
                steps_per_episode = steps_per_episode_counter

                total_rewards.append(ep_reward)
                mean_reward = np.mean(total_rewards[-100:])
                mean_rewards.append(mean_reward)
                step_indexes.append(step_counter)
                losses.append(loss)

                steps_per_episode_counter = 0
                ep_reward = 0.0

            else:
                batch_dones.append(False)
                batch_last_states.append(exp.last_state)

        last_states_t = torch.tensor(batch_last_states, dtype=torch.float32).to(device)
        states_t = torch.tensor(batch_states, dtype=torch.float32).to(device)
        actions_t = torch.tensor(batch_actions, dtype=torch.int64).to(device)

        with torch.no_grad():
            values_last = net(last_states_t)[1].squeeze() * GAMMA ** N_STEPS
        values_last[batch_dones] = 0  # the value on terminal states is zero!
        # Normalize the rewards
        values_target_t = torch.tensor(batch_values, dtype=torch.float32)
        std, mean = torch.std_mean(values_target_t)
        values_target_t -= mean
        values_target_t /= std + 1e-6

        optimizer.zero_grad()
        logits_t, values_t = net(states_t)
        values_t = values_t.squeeze()
        loss_v_t = F.mse_loss(values_t.squeeze(), values_target_t)

        log_prob_actions_t = F.log_softmax(logits_t, dim=1)
        log_probs_chosen = log_prob_actions_t.gather(
            1, actions_t.unsqueeze(1)
        ).squeeze()
        advantage_t = values_target_t - values_t.detach()
        loss_policy_t = (log_probs_chosen * advantage_t).mean()

        prob_actions_t = F.softmax(logits_t, dim=1)
        entropy_t = -(prob_actions_t * log_prob_actions_t).sum(dim=1).mean()
        loss_entropy_t = ENTROPY_BETA * entropy_t

        loss_total_t = -loss_policy_t + loss_v_t + loss_entropy_t
        loss_total_t.backward()
        optimizer.step()
        loss = loss_total_t.item()

        if step_counter % 200 == 0:
            print(
                f"{step_counter}: loss={loss:.6f}, ",
                f"mean reward={mean_reward:.2f}",
                f"steps per episode={steps_per_episode:.1f}, " f"episodes={ep_counter}",
            )
            env.render()

        # if mean_reward > 195:
        #     print(f"Solved in {step_counter} steps and {ep_counter} episodes")
        #     break

    # plt.plot(step_indexes, mean_rewards, label="mean reward")
    # plt.legend()
    # plt.show()
    # plt.plot(step_indexes, losses, label="loss")
    # plt.legend()
    # plt.show()