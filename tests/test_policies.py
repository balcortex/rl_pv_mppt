from src.policies import CategoricalDistributionPolicy, RandomPolicy, GreedyPolicy
import numpy as np
import torch


def test_categorical_single_out_net():
    net = torch.nn.Identity()
    device = torch.device("cpu")
    policy = CategoricalDistributionPolicy(net, device, apply_softmax=True)
    a = np.array([1000, 1, 1])
    action = policy(a, add_batch_dim=True)
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
    policy = CategoricalDistributionPolicy(net, device, apply_softmax=True, net_index=0)
    a = np.array([[1000, 1, 1], [1, 1, 1000]])
    action = policy(a, add_batch_dim=False)
    assert np.array_equal(action, np.array([0, 2]))


def test_random_single_out_net():
    net = torch.nn.Identity()
    device = torch.device("cpu")
    policy = RandomPolicy(net, device)
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
    policy = RandomPolicy(net, device, net_index=0)
    a = np.array([[1000, 1, 1], [1, 1, 1000]])
    action = policy(a, add_batch_dim=False)
    assert len(action) == 2


def test_greedy_single_out_net():
    net = torch.nn.Identity()
    device = torch.device("cpu")
    policy = GreedyPolicy(net, device)
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
    policy = GreedyPolicy(net, device, net_index=0)
    a = np.array([[1000, 1, 1], [1, 1, 1000]])
    action = policy(a)
    assert np.array_equal(action, np.array([0, 2]))