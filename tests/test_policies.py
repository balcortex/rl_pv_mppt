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
    policy = DiscreteCategoricalDistributionPolicy(
        net, device, apply_softmax=True, add_batch_dim=True
    )
    a = np.array([1000, 1, 1])
    action = policy(a)
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
        net, device, apply_softmax=True, net_index=0, add_batch_dim=False
    )
    a = np.array([[1000, 1, 1], [1, 1, 1000]])
    action = policy(a)
    assert np.array_equal(action, np.array([0, 2]))


def test_random_single_out_net():
    net = torch.nn.Identity()
    device = torch.device("cpu")
    policy = DiscreteRandomPolicy(net, device, add_batch_dim=True)
    a = np.array([1000, 1, 1])
    action = policy(a)
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
    policy = DiscreteRandomPolicy(net, device, net_index=0, add_batch_dim=False)
    a = np.array([[1000, 1, 1], [1, 1, 1000]])
    action = policy(a)
    assert len(action) == 2


def test_greedy_single_out_net():
    net = torch.nn.Identity()
    device = torch.device("cpu")
    policy = DiscreteGreedyPolicy(net, device, add_batch_dim=True)
    a = np.array([1000, 1, 1])
    action = policy(a)
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


def test_gaussian_policy_single_obs():
    class MyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(2, 1)

        def __call__(self, x):
            return (self.net(x),) * 3

    net = MyNet()
    device = torch.device("cpu")
    policy = GaussianPolicy(net, device, add_batch_dim=True)
    states = np.array([1.0, 2.0])
    action = policy(states)
    assert action.size == 1


def test_gaussian_policy_single_obs_test():
    class MyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.modules.Identity()

        def __call__(self, x):
            return self.net(x), self.net(x), self.net(x)

    net = MyNet()
    device = torch.device("cpu")
    policy = GaussianPolicy(net, device, add_batch_dim=True, test=True)
    states = np.array([1.0])
    action = policy(states)
    assert action == 1.0


def test_gaussian_policy_multi_obs():
    class MyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.modules.Identity()

        def __call__(self, x):
            return (self.net(x),) * 3

    net = MyNet()
    device = torch.device("cpu")
    policy = GaussianPolicy(net, device)
    states = np.array([[1.0], [2.0], [3.0]])
    action = policy(states)
    assert action.shape == (3, 1)


def test_gaussian_policy_multi_obs_test():
    class MyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.modules.Identity()

        def __call__(self, x):
            return (self.net(x),) * 3

    net = MyNet()
    device = torch.device("cpu")
    policy = GaussianPolicy(net, device, test=True)
    states = np.array([[1.0], [2.0], [3.0]])
    action = policy(states)
    assert np.array_equal(action, np.array([[1.0], [2.0], [3.0]]))
