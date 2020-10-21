import collections
import gym
from agents import BaseAgent

Experience = collections.namedtuple(
    "Experience", ["state", "action", "reward", "last_state"]
)
ExperienceDiscounted = collections.namedtuple(
    "Experience",
    ["state", "action", "reward", "last_state", "discounted_reward", "steps"],
)


class ExperienceSorce:
    def __init__(self, env: gym.Env, agent: BaseAgent):
        self.env = env
        self.agent = agent
        self.obs = self.env.reset()

    def __iter__(self):
        return self

    def __next__(self):
        return self._play_step()

    def _play_step(self):
        obs = self.obs
        action = int(self.agent(obs, training=False)[0])
        new_obs, reward, done, _ = self.env.step(action)
        if done:
            self.obs = self.env.reset()
            return Experience(state=obs, action=action, reward=reward, last_state=None)
        self.obs = new_obs
        return Experience(state=obs, action=action, reward=reward, last_state=new_obs)

    def _play_episode(self):
        ep_history = []
        self.obs = self.env.reset()

        while True:
            experience = self._play_step()
            ep_history.append(experience)

            if experience.last_state is None:
                return ep_history

    def _play_episodes(self, episodes):
        return [self._play_episode() for _ in range(episodes)]


class ExperienceSorceEpisodes(ExperienceSorce):
    def __init__(self, env: gym.Env, agent: BaseAgent, episodes: int):
        super().__init__(env, agent)

        self.max_episodes = episodes

    def __next__(self):
        return self._play_episodes(self.max_episodes)


class ExperienceSorceDiscounted(ExperienceSorce):
    def __init__(self, env: gym.Env, agent: BaseAgent, gamma: float, n_steps: int):
        super().__init__(env, agent)

        self.gamma = gamma
        self.max_steps = n_steps

    def __next__(self):
        return self._play_n_steps()

    def _play_n_steps(self):
        history = []
        discounted_reward = 0.0
        reward = 0.0

        for step_idx in range(self.max_steps):
            exp = self._play_step()
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

    def _play_episode(self):
        ep_history = []
        self.obs = self.env.reset()

        while True:
            experience = self._play_n_steps()
            ep_history.append(experience)

            if experience.last_state is None:
                return ep_history


class ExperienceSorceDiscountedSteps(ExperienceSorceDiscounted):
    def __init__(
        self,
        env: gym.Env,
        agent: BaseAgent,
        gamma: float,
        n_steps: int,
        steps: int,
    ):
        super().__init__(env, agent, gamma, n_steps)

        self.steps = steps

    def __next__(self):
        return [self._play_n_steps() for _ in range(self.steps)]


class ExperienceSorceDiscountedEpisodes(ExperienceSorceDiscounted):
    def __init__(
        self,
        env: gym.Env,
        agent: BaseAgent,
        gamma: float,
        n_steps: int,
        episodes: int,
    ):
        super().__init__(env, agent, gamma, n_steps)

        self.max_episodes = episodes

    def __next__(self):
        return self._play_episodes(self.max_episodes)


# class ExperienceSorce:
#     def __init__(self, env: gym.Env, agent: BaseAgent):
#         self.env = env
#         self.agent = agent
#         self.obs = self.env.reset()

#     def __iter__(self):
#         return self

#     def __next__(self):
#         return self._play_step()

#     def _play_step(self):
#         obs = self.obs
#         action = int(self.agent(obs, training=False)[0])
#         new_obs, reward, done, _ = self.env.step(action)
#         self.obs = self.env.reset() if done else new_obs
#         return Experience(state=obs, action=action, reward=reward, done=done)

#     def _play_episode(self):
#         ep_history = []
#         self.obs = self.env.reset()

#         while True:
#             experience = self._play_step()
#             ep_history.append(experience)

#             if experience.done:
#                 return ep_history

#     def _play_episodes(self, episodes):
#         return [self._play_episode() for _ in range(episodes)]


# class ExperienceSorceEpisodes(ExperienceSorce):
#     def __init__(self, env: gym.Env, agent: BaseAgent, episodes: int):
#         super().__init__(env, agent)

#         self.max_episodes = episodes

#     def __next__(self):
#         return self._play_episodes(self.max_episodes)


# class ExperienceSorceDiscounted(ExperienceSorce):
#     def __init__(self, env: gym.Env, agent: BaseAgent, gamma: float, steps_count: int):
#         super().__init__(env, agent)

#         self.gamma = gamma
#         self.max_steps = steps_count

#     def __next__(self):
#         return self._play_n_steps()

#     def _play_n_steps(self):
#         history = []
#         discounted_reward = 0.0
#         reward = 0.0

#         for step_idx in range(self.max_steps):
#             exp = self._play_step()
#             reward += exp.reward
#             discounted_reward += exp.reward * self.gamma ** (step_idx)
#             history.append(exp)

#             if exp.done:
#                 break

#         return ExperienceDiscounted(
#             state=history[0].state,
#             action=history[0].action,
#             reward=reward,
#             discounted_reward=discounted_reward,
#             steps=step_idx + 1,
#             done=history[-1].done,
#         )


# class ExperienceSorceDiscountedEpisodes(ExperienceSorceDiscounted):
#     def __init__(
#         self,
#         env: gym.Env,
#         agent: BaseAgent,
#         gamma: float,
#         steps_count: int,
#         episodes: int,
#     ):
#         super().__init__(env, agent, gamma, steps_count)

#         self.max_episodes = episodes

#     def __next__(self):
#         return self._play_episodes(self.max_episodes)

#     def _play_episode(self):
#         ep_history = []
#         self.obs = self.env.reset()

#         while True:
#             experience = self._play_n_steps()
#             ep_history.append(experience)

#             if experience.done:
#                 return ep_history
