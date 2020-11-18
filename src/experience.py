import collections
import gym
from src.policies import BasePolicy

Experience = collections.namedtuple(
    "Experience", ["state", "action", "reward", "last_state"]
)
ExperienceDiscounted = collections.namedtuple(
    "Experience",
    ["state", "action", "reward", "last_state", "discounted_reward", "steps"],
)


class ExperienceSorce:
    def __init__(self, env: gym.Env, policy: BasePolicy):
        self.env = env
        self.policy = policy
        self.obs = self.env.reset()
        self.done = False
        self.render = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.play_step()

    def play_step(self):
        if self.done:
            self.obs = self.env.reset()
            self.done = False
        obs = self.obs
        action = self.policy(obs)
        new_obs, reward, done, _ = self.env.step(action)
        if self.render:
            self.env.render()
        if done:
            self.done = True
            return Experience(state=obs, action=action, reward=reward, last_state=None)
        self.obs = new_obs
        return Experience(state=obs, action=action, reward=reward, last_state=new_obs)

    def play_episode(self):
        ep_history = []
        self.obs = self.env.reset()

        while True:
            experience = self.play_step()
            ep_history.append(experience)

            if experience.last_state is None:
                return ep_history

    def play_episodes(self, episodes):
        return [self.play_episode() for _ in range(episodes)]


class ExperienceSorceEpisodes(ExperienceSorce):
    def __init__(self, env: gym.Env, policy: BasePolicy, episodes: int):
        super().__init__(env, policy)

        self.max_episodes = episodes

    def __next__(self):
        return self.play_episodes(self.max_episodes)


class ExperienceSorceDiscounted(ExperienceSorce):
    def __init__(self, env: gym.Env, policy: BasePolicy, gamma: float, n_steps: int):
        super().__init__(env, policy)

        self.gamma = gamma
        self.max_steps = n_steps

    def __next__(self):
        return self.play_n_steps()

    def play_n_steps(self):
        history = []
        discounted_reward = 0.0
        reward = 0.0

        for step_idx in range(self.max_steps):
            exp = self.play_step()
            reward += exp.reward
            discounted_reward += exp.reward * self.gamma ** (step_idx)
            history.append(exp)

            if exp.last_state is None:
                break

        return ExperienceDiscounted(
            state=history[0].state,
            action=history[0].action,
            last_state=history[-1].last_state,
            reward=reward,
            discounted_reward=discounted_reward,
            steps=step_idx + 1,
        )

    def play_episode(self):
        ep_history = []
        self.obs = self.env.reset()

        while True:
            experience = self.play_n_steps()
            ep_history.append(experience)

            if experience.last_state is None:
                return ep_history


class ExperienceSorceDiscountedSteps(ExperienceSorceDiscounted):
    def __init__(
        self,
        env: gym.Env,
        policy: BasePolicy,
        gamma: float,
        n_steps: int,
        steps: int,
    ):
        super().__init__(env, policy, gamma, n_steps)

        self.steps = steps

    def __next__(self):
        return [self.play_n_steps() for _ in range(self.steps)]


class ExperienceSorceDiscountedEpisodes(ExperienceSorceDiscounted):
    def __init__(
        self,
        env: gym.Env,
        policy: BasePolicy,
        gamma: float,
        n_steps: int,
        episodes: int,
    ):
        super().__init__(env, policy, gamma, n_steps)

        self.max_episodes = episodes

    def __next__(self):
        return self.play_episodes(self.max_episodes)
