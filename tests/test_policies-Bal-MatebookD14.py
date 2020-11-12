from src.policies import (
    DiscreteCategoricalDistributionPolicy,
    DiscreteRandomPolicy,
    DiscreteGreedyPolicy,
    GaussianPolicy,
)
import numpy as np
import torch


def test_categorical_single_out_net():
    net = torch.nn.Identity()
    device = torch.device("cpu")
    policy = DiscreteCategoricalDistributionPolicy(net, device, apply_softmax=True)
    a = np.array([1000, 1, 1])
    action = policy(a, add_batch_dim=False)
    assert action == 0


def test_categorical_double_out_net():
    class DoubleOutNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.id1 = torch.nn.Identity()

        def __call__(self, x):
            return self.id1(x), torch.tensor(1)

    net = DoubleOutNet()
    device = torch.device("cpu")
    policy = DiscreteCategoricalDistributionPolicy(
        net, device, apply_softmax=True, net_index=0
    )
    a = np.array([[1000, 1, 1], [1, 1, 1000]])
    action = policy(a, add_batch_dim=False)
    assert np.array_equal(action, np.array([0, 2]))


def test_random_single_out_net():
    net = torch.nn.Identity()
    device = torch.device("cpu")
    policy = DiscreteRandomPolicy(net, device)
    a = np.array([1000, 1, 1])
    action = policy(a, add_batch_dim=True)
    assert not isinstance(action, np.ndarray)


def test_random_double_out_net():
    class DoubleOutNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.id1 = torch.nn.Identity()

        def __call__(self, x):
            return self.id1(x), torch.tensor(1)

    net = DoubleOutNet()
    device = torch.device("cpu")
    policy = DiscreteRandomPolicy(net, device, net_index=0)
    a = np.array([[1000, 1, 1], [1, 1, 1000]])
    action = policy(a, add_batch_dim=False)
    assert len(action) == 2


def test_greedy_single_out_net():
    net = torch.nn.Identity()
    device = torch.device("cpu")
    policy = DiscreteGreedyPolicy(net, device)
    a = np.array([1000, 1, 1])
    action = policy(a, add_batch_dim=True)
    assert action == [0]


def test_greedy_double_out_net():
    class DoubleOutNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.id1 = torch.nn.Identity()

        def __call__(self, x):
            return self.id1(x), torch.tensor(1)

    net = DoubleOutNet()
    device = torch.device("cpu")
    policy = DiscreteGreedyPolicy(net, device, net_index=0)
    a = np.array([[1000, 1, 1], [1, 1, 1000]])
    action = policy(a)
    assert np.array_equal(action, np.array([0, 2]))


def test_gaussian_policy():
    class MyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(1, 1)

        def __call__(self, x):
            return (self.net(x),) * 3

    net = MyNet()
    device = torch.device("cpu")
    policy = GaussianPolicy(net, device)
    states = np.array([[1.0], [2.0]])
    # obs = np.array([1.0])
    action = policy(obs, add_batch_dim=False)
    action

    states = np.array([[1.0], [2.0]])
    states = np.array([1.0])
    states_v = torch.tensor(states, dtype=torch.float32).to(device)
    mean_t, std_t, _ = net(states_v)
    actions = torch.normal(mean_t, std_t)
    actions.clamp(-1, 1).squeeze()