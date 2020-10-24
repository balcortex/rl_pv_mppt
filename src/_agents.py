import torch
import numpy as np


class BaseAgent:
    @torch.no_grad()
    def __call__(self, states, training=True):
        raise NotImplementedError


class PolicyAgent:
    def __init__(self, net, device="cpu", apply_softmax=False):
        self.net = net
        self.device = device
        self.apply_softmax = apply_softmax

    @torch.no_grad()
    def __call__(self, states, greedy=False):
        states_v = torch.tensor(states, dtype=torch.float32).to(self.device)
        probs_t = self.net(states_v)
        if greedy:
            return torch.argmax(probs_t, dim=1).cpu().numpy()
        if self.apply_softmax:
            probs_t = torch.nn.functional.softmax(probs_t, dim=1)
        probs_dist = torch.distributions.Categorical(probs_t)
        return probs_dist.sample().cpu().numpy()

    @staticmethod
    def choose_action_numpy(probs):
        actions = range(len(probs[0]))
        return np.array([np.random.choice(actions, p=prob) for prob in probs])


class ActorCritic(PolicyAgent):
    @torch.no_grad()
    def __call__(self, states, training=True):
        if not training:
            states = [states]  # add batch dimension
        states_v = torch.tensor(states, dtype=torch.float32).to(self.device)
        probs_t, _ = self.net(states_v)
        if self.apply_softmax:
            probs_t = torch.nn.functional.softmax(probs_t, dim=1)
        probs_dist = torch.distributions.Categorical(probs_t)
        return probs_dist.sample().cpu().numpy()


if __name__ == "__main__":
    states = [[20, 1, 1], [1, 1.5, 1], [1, 1, 2]]

    net = torch.nn.Identity()
    pa = PolicyAgent(net, apply_softmax=True)
    actions = pa(states)
    greedy_actions = pa(states, greedy=True)

    print("Sampled actions")
    print(actions, end="\n")
    print("Greedy actions")
    print(greedy_actions)
